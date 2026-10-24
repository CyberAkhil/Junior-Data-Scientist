from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from week3_feature_engineering import engineer_batch_features
from week5_model_evaluation import classification_metrics, chronological_split_indices


RANDOM_STATE = 42
IDENTIFIER_COLUMNS = {"batch_id", "prediction_cutoff", "cluster"}


def _validate_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_columns: list[str],
    group_column: str | None,
) -> pd.DataFrame:
    required = {"batch_id", *target_columns}
    if group_column:
        required.add(group_column)
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"Target data is missing columns: {sorted(missing)}")
    collisions = set(target_columns) & set(features.columns)
    if collisions:
        raise ValueError(
            f"Target columns cannot also be model features: {sorted(collisions)}"
        )
    if features["batch_id"].duplicated().any():
        raise ValueError("Features must contain one row per batch")
    if targets["batch_id"].duplicated().any():
        raise ValueError("Targets must contain one row per batch")
    if targets[target_columns].isna().any().any():
        raise ValueError("Target values cannot be missing")

    selected = ["batch_id", *target_columns]
    if group_column:
        if group_column in target_columns:
            raise ValueError("The group column cannot also be a prediction target")
        selected.append(group_column)
    merged = features.merge(
        targets[selected],
        on="batch_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "__target"),
    )
    if len(merged) != len(features) or len(merged) != len(targets):
        raise ValueError("Features and targets must cover the same batches")
    return merged


def _feature_columns(
    data: pd.DataFrame,
    group_column: str | None,
) -> list[str]:
    excluded = IDENTIFIER_COLUMNS | ({group_column} if group_column else set())
    columns = [
        column
        for column in data.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(data[column])
    ]
    if not columns:
        raise ValueError("No numeric batch features are available")
    values = data[columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Features must be finite or missing")
    return columns


def _chronological_partition(
    data: pd.DataFrame,
    holdout_fraction: float,
    group_column: str | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    train, test = chronological_split_indices(
        data["prediction_cutoff"], test_fraction=holdout_fraction
    )
    if len(train) < 2 or len(test) == 0:
        raise ValueError("Chronological holdout needs at least two training batches")
    if group_column:
        if group_column not in data.columns:
            raise ValueError(f"Unknown group column: {group_column}")
        groups = data[group_column]
        if groups.isna().any():
            raise ValueError(f"{group_column} cannot contain missing values")
        overlap = set(groups.iloc[train]) & set(groups.iloc[test])
        if overlap:
            raise ValueError(
                f"{len(overlap)} {group_column} group(s) cross the chronological "
                "boundary; choose a production cutoff that keeps groups together"
            )
    test_start = str(data["prediction_cutoff"].iloc[test].min())
    return train, test, test_start


def _classifier_models() -> dict[str, Any]:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
        "decision_tree": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def _regressor_models() -> dict[str, Any]:
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "decision_tree": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            DecisionTreeRegressor(
                max_depth=4,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def evaluate_quality_models(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    binary_target: str | None = None,
    continuous_target: str | None = None,
    holdout_fraction: float = 0.2,
    threshold: float = 0.5,
    group_column: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare baseline and candidate models on a chronological batch holdout."""
    if binary_target is None and continuous_target is None:
        raise ValueError("Specify a binary target, continuous target, or both")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not {"batch_id", "prediction_cutoff"}.issubset(features.columns):
        raise ValueError("Features must include batch_id and prediction_cutoff")
    target_columns = [
        column for column in (binary_target, continuous_target) if column is not None
    ]
    data = _validate_targets(features, targets, target_columns, group_column)
    columns = _feature_columns(data, group_column)
    train, test, test_start = _chronological_partition(
        data, holdout_fraction, group_column
    )
    x_train = data.iloc[train][columns]
    x_test = data.iloc[test][columns]
    report: dict[str, Any] = {
        "train_batches": int(len(train)),
        "test_batches": int(len(test)),
        "test_start": test_start,
        "features": columns,
        "decision_threshold": threshold if binary_target else None,
        "classification": {},
        "regression": {},
    }
    prediction_rows = []
    ids = data.iloc[test]["batch_id"].to_numpy()
    cutoffs = data.iloc[test]["prediction_cutoff"].astype(str).to_numpy()

    if binary_target is not None:
        labels = pd.to_numeric(data[binary_target], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{binary_target} must contain only binary values 0 and 1")
        y = labels.to_numpy(dtype=int)
        y_train, y_test = y[train], y[test]
        if np.unique(y_train).size != 2:
            raise ValueError(
                "Chronological training partition must contain both target classes"
            )
        positive_rate = float(y_train.mean())
        baseline_probability = np.full(len(test), positive_rate)
        baseline_prediction = np.full(
            len(test), int(positive_rate >= threshold), dtype=int
        )
        candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "prevalence_baseline": (baseline_prediction, baseline_probability)
        }
        for name, model in _classifier_models().items():
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_test)[:, 1]
            prediction = (probability >= threshold).astype(int)
            candidates[name] = (prediction, probability)

        report["classification"]["training_prevalence"] = positive_rate
        report["classification"]["models"] = {}
        for name, (prediction, probability) in candidates.items():
            scores = classification_metrics(
                y_test, prediction, probability, positive_label=1
            )
            report["classification"]["models"][name] = scores
            prediction_rows.extend(
                {
                    "batch_id": batch_id,
                    "prediction_cutoff": cutoff,
                    "task": "classification",
                    "model": name,
                    "actual": int(actual),
                    "predicted": int(estimated),
                    "probability": float(probability_value),
                }
                for batch_id, cutoff, actual, estimated, probability_value in zip(
                    ids, cutoffs, y_test, prediction, probability
                )
            )

    if continuous_target is not None:
        values = pd.to_numeric(data[continuous_target], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{continuous_target} must contain finite numeric values")
        y = values.to_numpy(dtype=float)
        y_train, y_test = y[train], y[test]
        candidates_regression: dict[str, np.ndarray] = {
            "median_baseline": np.full(len(test), float(np.median(y_train)))
        }
        for name, model in _regressor_models().items():
            model.fit(x_train, y_train)
            candidates_regression[name] = model.predict(x_test)

        report["regression"]["models"] = {}
        for name, prediction in candidates_regression.items():
            scores = {
                "mae": float(mean_absolute_error(y_test, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                "bias": float(np.mean(prediction - y_test)),
                "r2": (
                    float(r2_score(y_test, prediction))
                    if len(y_test) > 1
                    else None
                ),
            }
            report["regression"]["models"][name] = scores
            prediction_rows.extend(
                {
                    "batch_id": batch_id,
                    "prediction_cutoff": cutoff,
                    "task": "regression",
                    "model": name,
                    "actual": float(actual),
                    "predicted": float(estimated),
                    "probability": None,
                }
                for batch_id, cutoff, actual, estimated in zip(
                    ids, cutoffs, y_test, prediction
                )
            )

    return report, pd.DataFrame(prediction_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate food-quality models on a chronological batch holdout."
    )
    parser.add_argument("readings_csv", type=Path)
    parser.add_argument("targets_csv", type=Path)
    parser.add_argument("--binary-target")
    parser.add_argument("--continuous-target")
    parser.add_argument("--group-column")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Candidate threshold; select it using validation data, not the final test.",
    )
    parser.add_argument("--max-gap-minutes", type=float, default=15.0)
    parser.add_argument("--predictions-output", type=Path, default=Path("week4_predictions.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("week4_model_report.json"))
    args = parser.parse_args()
    if args.binary_target is None and args.continuous_target is None:
        parser.error("provide --binary-target and/or --continuous-target")

    features = engineer_batch_features(
        pd.read_csv(args.readings_csv),
        max_gap_minutes=args.max_gap_minutes,
    )
    targets = pd.read_csv(args.targets_csv)
    report, predictions = evaluate_quality_models(
        features,
        targets,
        binary_target=args.binary_target,
        continuous_target=args.continuous_target,
        holdout_fraction=args.holdout_fraction,
        threshold=args.threshold,
        group_column=args.group_column,
    )
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)
    args.report_output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote holdout predictions to {args.predictions_output}")
    print(f"Wrote model report to {args.report_output}")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
