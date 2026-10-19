from __future__ import annotations

import argparse
import warnings
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABELS = {
    0: "FACILITY CHANGED",
    1: "FAIL",
    2: "FURTHER INSPECTION REQUIRED",
    3: "INSPECTION OVERRULED",
    4: "PASS",
    5: "PASS(CONDITIONAL)",
    6: "SHUT-DOWN",
}
PRE_INSPECTION_CATEGORICAL = ["FacilityID", "Type", "Reason"]
ADDITIONAL_CANDIDATE_FIELDS = ["RiskLevel", "SectionViolations"]
DATE_FEATURES = ["Year", "Month", "DayOfWeek"]
RANDOM_STATE = 42


def load_workbooks(archive_path: Path):
    with ZipFile(archive_path) as archive:
        train = pd.read_excel(
            BytesIO(archive.read("Data_Train.xlsx")), engine="openpyxl"
        )
        test = pd.read_excel(
            BytesIO(archive.read("Data_Test.xlsx")), engine="openpyxl"
        )
        sample = pd.read_excel(
            BytesIO(archive.read("Sample_Submission.xlsx")), engine="openpyxl"
        )

    if "Inspection_Results" not in train.columns:
        raise ValueError("Data_Train.xlsx must contain Inspection_Results")
    if len(test) != len(sample):
        raise ValueError("Data_Test.xlsx and Sample_Submission.xlsx row counts differ")

    expected_test_columns = set(train.columns) - {"Inspection_Results"}
    if expected_test_columns != set(test.columns):
        raise ValueError("Train and test predictor columns do not match")

    return train, test, sample


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    dates = []

    for frame in (train, test):
        parsed = pd.to_datetime(frame["Date"], format="%d-%m-%Y", errors="coerce")
        dates.append(parsed)
        frame["Year"] = parsed.dt.year
        frame["Month"] = parsed.dt.month
        frame["DayOfWeek"] = parsed.dt.dayofweek

    categorical = PRE_INSPECTION_CATEGORICAL + ADDITIONAL_CANDIDATE_FIELDS
    for column in categorical:
        train[column] = train[column].astype("string").fillna("Missing")
        test[column] = test[column].astype("string").fillna("Missing")

    return train, test, dates


def build_model(categorical_columns: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=15,
                ),
                categorical_columns,
            ),
            (
                "date",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                DATE_FEATURES,
            ),
        ]
    )
    classifier = LogisticRegression(
        C=0.5,
        max_iter=180,
        tol=0.003,
        solver="saga",
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("classifier", classifier),
        ]
    )


def report_scores(
    name: str,
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    labels = np.sort(target.unique())

    print(f"\n{name} (n={len(target):,})")
    print(f"Accuracy:          {accuracy_score(target, predictions):.5f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(target, predictions):.5f}"
    )
    print(
        "Macro-F1:          "
        f"{f1_score(target, predictions, average='macro', zero_division=0):.5f}"
    )
    print(
        "Weighted F1:       "
        f"{f1_score(target, predictions, average='weighted', zero_division=0):.5f}"
    )
    print(
        f"Log loss:          "
        f"{log_loss(target, probabilities, labels=model.classes_):.5f}"
    )
    print(
        classification_report(
            target,
            predictions,
            labels=labels,
            target_names=[LABELS[int(label)] for label in labels],
            zero_division=0,
        )
    )
    return predictions


def profile_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_missing_train: pd.Series,
    source_missing_test: pd.Series,
    train_dates: pd.Series,
    test_dates: pd.Series,
    target: pd.Series,
) -> None:
    print("Training shape:", train.shape)
    print("Test shape:", test.shape)
    print("Target counts:", target.value_counts().sort_index().to_dict())
    print("Source training missing values:", source_missing_train.to_dict())
    print("Source test missing values:", source_missing_test.to_dict())
    print(
        "Unparseable dates (train/test):",
        int(train_dates.isna().sum()),
        int(test_dates.isna().sum()),
    )
    print("Unique training facilities:", train["FacilityID"].nunique())
    print("Duplicate training IDs:", int(train["ID"].duplicated().sum()))
    print("Duplicate test IDs:", int(test["ID"].duplicated().sum()))


def validate_models(train: pd.DataFrame, target: pd.Series, train_dates: pd.Series):
    train_rows, validation_rows = train_test_split(
        np.arange(len(train)),
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    majority_class = target.iloc[train_rows].value_counts().idxmax()
    majority_predictions = np.full(len(validation_rows), majority_class)
    random_target = target.iloc[validation_rows]
    print("\nRandom validation majority baseline")
    print(
        f"Accuracy: {accuracy_score(random_target, majority_predictions):.5f}"
    )
    print(
        "Macro-F1:  "
        f"{f1_score(random_target, majority_predictions, average='macro', zero_division=0):.5f}"
    )

    feature_sets = {
        "Random holdout, pre-inspection fields": PRE_INSPECTION_CATEGORICAL,
        "Random holdout, all supplied candidate fields": (
            PRE_INSPECTION_CATEGORICAL + ADDITIONAL_CANDIDATE_FIELDS
        ),
    }
    for name, categorical_columns in feature_sets.items():
        model = build_model(categorical_columns)
        model.fit(train.iloc[train_rows], target.iloc[train_rows])
        report_scores(name, model, train.iloc[validation_rows], random_target)

    training_period = (train_dates.dt.year < 2017).fillna(False)
    validation_period = (train_dates.dt.year >= 2017).fillna(False)
    past_target = target.loc[training_period]
    future_target = target.loc[validation_period]
    past_majority = past_target.value_counts().idxmax()
    past_baseline = np.full(len(future_target), past_majority)

    print("\nChronological validation: train through 2016, validate on 2017-2018")
    print("Invalid training dates excluded:", int(train_dates.isna().sum()))
    print(
        "Historical majority baseline accuracy: "
        f"{accuracy_score(future_target, past_baseline):.5f}"
    )
    print(
        "Historical majority baseline macro-F1: "
        f"{f1_score(future_target, past_baseline, average='macro', zero_division=0):.5f}"
    )

    for name, categorical_columns in feature_sets.items():
        model = build_model(categorical_columns)
        model.fit(train.loc[training_period], past_target)
        temporal_name = name.replace("Random holdout", "Temporal holdout")
        report_scores(
            temporal_name,
            model,
            train.loc[validation_period],
            future_target,
        )


def summarize_test_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target: pd.Series,
) -> None:
    expected_headers = [LABELS[int(label)] for label in sorted(target.unique())]
    if list(sample.columns) != expected_headers:
        raise ValueError(
            "Sample-submission headers do not match the inferred target mapping: "
            f"{list(sample.columns)}"
        )

    model = build_model(
        PRE_INSPECTION_CATEGORICAL + ADDITIONAL_CANDIDATE_FIELDS
    )
    model.fit(train, target)
    probabilities = model.predict_proba(test)
    predictions = model.classes_[probabilities.argmax(axis=1)]

    print("\nTest prediction summary (no submission file is written)")
    print("Probability shape:", probabilities.shape)
    print(
        "Probability row-sum range:",
        float(probabilities.sum(axis=1).min()),
        float(probabilities.sum(axis=1).max()),
    )
    print(
        "Predicted outcome counts:",
        Counter(LABELS[int(value)] for value in predictions),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore the supplied food-inspection dataset and evaluate a baseline."
    )
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent
        / "data"
        / "food_inspection_dataset.zip",
        help="Path to the ZIP containing the three Excel workbooks.",
    )
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(
            f"Dataset archive not found: {args.archive}. "
            "Pass its path as the first command-line argument."
        )

    raw_train, raw_test, sample = load_workbooks(args.archive)
    source_missing_train = raw_train.isna().sum()
    source_missing_test = raw_test.isna().sum()
    train, test, parsed_dates = prepare_features(raw_train, raw_test)
    target = train["Inspection_Results"].astype(int)

    profile_data(
        train,
        test,
        source_missing_train,
        source_missing_test,
        parsed_dates[0],
        parsed_dates[1],
        target,
    )
    validate_models(train, target, parsed_dates[0])
    summarize_test_predictions(train, test, sample, target)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    main()
