from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


REQUIRED_COLUMNS = {
    "batch_id",
    "timestamp",
    "prediction_cutoff",
    "sensor",
    "value",
}
OPTIONAL_NUMERIC_COLUMNS = {"target", "lower_limit", "upper_limit"}
RANDOM_STATE = 42


def engineer_batch_features(
    readings: pd.DataFrame,
    max_gap_minutes: float = 15.0,
) -> pd.DataFrame:
    """Summarize long-form sensor readings available by each batch's cutoff.

    Optional target and limit columns must be expressed in the same units as
    value. Exposure durations use a forward-hold assumption only between
    adjacent readings no farther apart than max_gap_minutes.
    """
    missing = REQUIRED_COLUMNS - set(readings.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not np.isfinite(max_gap_minutes) or max_gap_minutes <= 0:
        raise ValueError("max_gap_minutes must be a positive finite number")
    if readings.empty:
        raise ValueError("Readings contain no rows")

    data = readings.copy()
    for column in ("batch_id", "sensor", "timestamp", "prediction_cutoff"):
        if data[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["prediction_cutoff"] = pd.to_datetime(
        data["prediction_cutoff"], errors="coerce"
    )
    if data["timestamp"].isna().any() or data["prediction_cutoff"].isna().any():
        raise ValueError("Timestamps and prediction cutoffs must be valid")
    if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
        raise ValueError("Timestamps must use one consistent timezone")
    if not pd.api.types.is_datetime64_any_dtype(data["prediction_cutoff"]):
        raise ValueError("Prediction cutoffs must use one consistent timezone")

    value = pd.to_numeric(data["value"], errors="coerce")
    invalid_values = data["value"].notna() & value.isna()
    if invalid_values.any():
        raise ValueError("Non-numeric values must be corrected before feature engineering")
    if np.isinf(value.dropna()).any():
        raise ValueError("Sensor values must be finite or missing")
    data["value"] = value

    for column in OPTIONAL_NUMERIC_COLUMNS & set(data.columns):
        numeric = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"{column} contains non-numeric values")
        if np.isinf(numeric.dropna()).any():
            raise ValueError(f"{column} values must be finite or missing")
        data[column] = numeric

    if {"lower_limit", "upper_limit"}.issubset(data.columns):
        invalid_limits = (
            data["lower_limit"].notna()
            & data["upper_limit"].notna()
            & (data["lower_limit"] > data["upper_limit"])
        )
        if invalid_limits.any():
            raise ValueError("lower_limit cannot exceed upper_limit")

    cutoff_counts = data.groupby("batch_id", dropna=False)[
        "prediction_cutoff"
    ].nunique()
    if cutoff_counts.gt(1).any():
        raise ValueError("Each batch must have one prediction_cutoff")

    duplicate_events = data.duplicated(
        ["batch_id", "sensor", "timestamp"], keep=False
    )
    if duplicate_events.any():
        raise ValueError(
            "Duplicate batch/sensor/timestamp rows found; resolve replicates first"
        )

    data = data.loc[data["timestamp"] <= data["prediction_cutoff"]].copy()
    if data.empty:
        raise ValueError("No sensor readings are available by the prediction cutoffs")
    data = data.sort_values(
        ["batch_id", "sensor", "timestamp"], kind="stable"
    )

    sensor_names = data["sensor"].astype(str).unique()
    sensor_keys = {
        sensor: re.sub(r"[^0-9A-Za-z]+", "_", sensor).strip("_").lower()
        for sensor in sensor_names
    }
    if len(set(sensor_keys.values())) != len(sensor_keys):
        raise ValueError("Sensor names collide after conversion to feature names")

    batch_features: dict[object, dict[str, float | int | pd.Timestamp]] = {}
    for batch_id, batch_rows in data.groupby("batch_id", sort=False):
        features: dict[str, float | int | pd.Timestamp] = {
            "batch_id": batch_id,
            "prediction_cutoff": batch_rows["prediction_cutoff"].iloc[0],
        }
        for sensor, group in batch_rows.groupby("sensor", sort=False):
            group = group.sort_values("timestamp", kind="stable")
            key = sensor_keys[str(sensor)]
            prefix = f"{key}__"
            values = group["value"].to_numpy(dtype=float)
            valid = np.isfinite(values)
            valid_values = values[valid]
            timestamps = group["timestamp"]

            features[prefix + "reading_count"] = len(group)
            features[prefix + "valid_count"] = int(valid.sum())
            features[prefix + "missing_fraction"] = float(1 - valid.mean())
            if valid_values.size:
                features[prefix + "mean"] = float(valid_values.mean())
                features[prefix + "std"] = float(valid_values.std(ddof=0))
                features[prefix + "min"] = float(valid_values.min())
                features[prefix + "max"] = float(valid_values.max())
                features[prefix + "range"] = float(
                    valid_values.max() - valid_values.min()
                )
                features[prefix + "start"] = float(valid_values[0])
                features[prefix + "end"] = float(valid_values[-1])

                elapsed_minutes = (
                    timestamps - timestamps.iloc[0]
                ).dt.total_seconds().to_numpy() / 60.0
                valid_times = elapsed_minutes[valid]
                if valid.sum() >= 2 and np.ptp(valid_times) > 0:
                    centered_times = valid_times - valid_times.mean()
                    slope = np.dot(
                        centered_times,
                        valid_values - valid_values.mean(),
                    ) / np.dot(centered_times, centered_times)
                    features[prefix + "slope_per_minute"] = float(slope)

            if "target" in group.columns:
                targets = group["target"].to_numpy(dtype=float)
                comparable = valid & np.isfinite(targets)
                if comparable.any():
                    deviations = values[comparable] - targets[comparable]
                    features[prefix + "mean_target_deviation"] = float(
                        deviations.mean()
                    )
                    features[prefix + "mean_absolute_target_deviation"] = float(
                        np.abs(deviations).mean()
                    )

            observed_minutes = 0.0
            interval_rows = []
            for index in range(len(group) - 1):
                gap = (
                    timestamps.iloc[index + 1] - timestamps.iloc[index]
                ).total_seconds() / 60.0
                if gap <= 0 or gap > max_gap_minutes or not valid[index]:
                    continue
                observed_minutes += gap
                interval_rows.append((index, gap))
            features[prefix + "observed_minutes"] = observed_minutes

            if {"lower_limit", "upper_limit"}.issubset(group.columns):
                lower = group["lower_limit"].to_numpy(dtype=float)
                upper = group["upper_limit"].to_numpy(dtype=float)
                has_spec = valid & np.isfinite(lower) & np.isfinite(upper)
                if has_spec.any():
                    in_spec = (values >= lower) & (values <= upper)
                    features[prefix + "below_limit_fraction"] = float(
                        (values[has_spec] < lower[has_spec]).mean()
                    )
                    features[prefix + "within_spec_fraction"] = float(
                        in_spec[has_spec].mean()
                    )
                    features[prefix + "above_limit_fraction"] = float(
                        (values[has_spec] > upper[has_spec]).mean()
                    )

                in_spec_minutes = 0.0
                out_of_spec_minutes = 0.0
                for index, gap in interval_rows:
                    if not (
                        np.isfinite(lower[index]) and np.isfinite(upper[index])
                    ):
                        continue
                    if lower[index] <= values[index] <= upper[index]:
                        in_spec_minutes += gap
                    else:
                        out_of_spec_minutes += gap
                features[prefix + "in_spec_minutes"] = in_spec_minutes
                features[prefix + "out_of_spec_minutes"] = out_of_spec_minutes

        batch_features[batch_id] = features

    return pd.DataFrame(batch_features.values()).reset_index(drop=True)


def add_polynomial_features(
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Add degree-two terms for an explicitly selected numeric feature subset."""
    if not columns:
        raise ValueError("Select at least one feature for polynomial expansion")
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"Unknown polynomial feature columns: {sorted(missing)}")
    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in columns):
        raise ValueError("Polynomial features must be numeric")

    transformer = PolynomialFeatures(degree=2, include_bias=False)
    expanded = transformer.fit_transform(features[columns])
    names = [
        "poly__" + name.replace(" ", "__")
        for name in transformer.get_feature_names_out(columns)
    ]
    collisions = set(names) & set(features.columns)
    if collisions:
        raise ValueError(f"Polynomial feature names already exist: {sorted(collisions)}")
    result = features.copy()
    result[names] = expanded
    return result


def add_interaction(
    features: pd.DataFrame,
    left: str,
    right: str,
) -> pd.DataFrame:
    """Add one domain-selected numeric interaction without expanding all pairs."""
    missing = {left, right} - set(features.columns)
    if missing:
        raise ValueError(f"Unknown interaction columns: {sorted(missing)}")
    if not all(
        pd.api.types.is_numeric_dtype(features[column])
        for column in (left, right)
    ):
        raise ValueError("Interaction features must be numeric")
    name = f"{left}__x__{right}"
    if name in features.columns:
        raise ValueError(f"Interaction feature already exists: {name}")
    result = features.copy()
    result[name] = result[left] * result[right]
    return result


def evaluate_regression(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_column: str = "quality_target",
    holdout_fraction: float = 0.2,
) -> pd.DataFrame:
    """Compare regression baselines using a chronological batch holdout."""
    required_targets = {"batch_id", target_column}
    missing = required_targets - set(targets.columns)
    if missing:
        raise ValueError(f"Target data is missing columns: {sorted(missing)}")
    if "batch_id" not in features or "prediction_cutoff" not in features:
        raise ValueError("Features must include batch_id and prediction_cutoff")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if targets["batch_id"].duplicated().any():
        raise ValueError("Targets must contain one row per batch")
    if features["batch_id"].duplicated().any():
        raise ValueError("Features must contain one row per batch")

    target_values = pd.to_numeric(targets[target_column], errors="coerce")
    if target_values.isna().any() or not np.isfinite(target_values).all():
        raise ValueError("Quality targets must be finite numeric values")
    merged = features.merge(
        targets[["batch_id"]].assign(**{target_column: target_values}),
        on="batch_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(targets):
        raise ValueError("Features and quality targets must cover the same batches")

    numeric_columns = [
        column
        for column in features.columns
        if column not in {"batch_id", "prediction_cutoff", "cluster"}
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not numeric_columns:
        raise ValueError("No numeric features are available for regression")
    if np.isinf(merged[numeric_columns].to_numpy(dtype=float)).any():
        raise ValueError("Regression features must be finite or missing")
    merged["prediction_cutoff"] = pd.to_datetime(
        merged["prediction_cutoff"], errors="coerce"
    )
    if merged["prediction_cutoff"].isna().any():
        raise ValueError("Every batch must have a valid prediction cutoff")

    cutoffs = np.sort(merged["prediction_cutoff"].unique())
    if len(cutoffs) < 2:
        raise ValueError("At least two distinct cutoff times are needed")
    split_index = int(np.floor(len(cutoffs) * (1 - holdout_fraction)))
    split_index = min(max(split_index, 1), len(cutoffs) - 1)
    boundary = cutoffs[split_index]
    train = merged["prediction_cutoff"] < boundary
    test = ~train
    if train.sum() < 2 or not test.any():
        raise ValueError("The chronological split needs at least two training batches")

    x_train = merged.loc[train, numeric_columns]
    x_test = merged.loc[test, numeric_columns]
    y_train = merged.loc[train, target_column]
    y_test = merged.loc[test, target_column]
    models = {
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
    }

    median_prediction = np.full(len(y_test), y_train.median())
    rows = [
        {
            "model": "median_baseline",
            "train_batches": int(train.sum()),
            "test_batches": int(test.sum()),
            "mae": mean_absolute_error(y_test, median_prediction),
            "rmse": float(np.sqrt(mean_squared_error(y_test, median_prediction))),
            "r2": (
                r2_score(y_test, median_prediction)
                if len(y_test) > 1
                else np.nan
            ),
            "test_start": str(boundary),
        }
    ]
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        rows.append(
            {
                "model": name,
                "train_batches": int(train.sum()),
                "test_batches": int(test.sum()),
                "mae": mean_absolute_error(y_test, prediction),
                "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                "r2": (
                    r2_score(y_test, prediction) if len(y_test) > 1 else np.nan
                ),
                "test_start": str(boundary),
            }
        )
    return pd.DataFrame(rows)


def cluster_batch_profiles(
    features: pd.DataFrame,
    n_clusters: int = 4,
) -> pd.DataFrame:
    """Return exploratory K-means labels for numeric batch features."""
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    numeric_columns = [
        column
        for column in features.columns
        if column not in {"batch_id", "prediction_cutoff"}
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not numeric_columns:
        raise ValueError("No numeric features are available for clustering")
    if len(features) < n_clusters:
        raise ValueError("There must be at least n_clusters batches")
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE),
    )
    labels = pipeline.fit_predict(features[numeric_columns])
    return pd.DataFrame({"batch_id": features["batch_id"], "cluster": labels})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Engineer leakage-aware batch features from long-form sensor data."
    )
    parser.add_argument("readings_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets-csv", type=Path)
    parser.add_argument("--target-column", default="quality_target")
    parser.add_argument("--max-gap-minutes", type=float, default=15.0)
    parser.add_argument("--clusters", type=int)
    parser.add_argument("--polynomial-features", nargs="+")
    parser.add_argument("--interaction", nargs=2, action="append", default=[])
    args = parser.parse_args()

    readings = pd.read_csv(args.readings_csv)
    features = engineer_batch_features(readings, args.max_gap_minutes)
    for left, right in args.interaction:
        features = add_interaction(features, left, right)
    if args.polynomial_features:
        features = add_polynomial_features(features, args.polynomial_features)
    if args.clusters is not None:
        labels = cluster_batch_profiles(features, args.clusters)
        features = features.merge(labels, on="batch_id", validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} batch feature rows to {args.output}")

    if args.targets_csv:
        targets = pd.read_csv(args.targets_csv)
        scores = evaluate_regression(
            features,
            targets,
            target_column=args.target_column,
        )
        print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
