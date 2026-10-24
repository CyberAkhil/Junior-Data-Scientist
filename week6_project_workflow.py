from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from week3_feature_engineering import engineer_batch_features
from week4_predictive_modeling import evaluate_quality_models
from week5_model_evaluation import bootstrap_confidence_intervals


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regression_intervals(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    samples: int,
    confidence: float,
    random_state: int,
) -> dict[str, dict[str, float | int]]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if pd.isna(groups).any():
        raise ValueError("Bootstrap groups cannot contain missing values")

    unique_groups = pd.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two independent test groups are required")
    group_rows = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {"mae": [], "rmse": [], "bias": []}
    for _ in range(samples):
        selected_groups = rng.integers(0, len(unique_groups), len(unique_groups))
        indices = np.concatenate([group_rows[index] for index in selected_groups])
        errors = predicted[indices] - actual[indices]
        values["mae"].append(float(np.mean(np.abs(errors))))
        values["rmse"].append(float(np.sqrt(np.mean(errors**2))))
        values["bias"].append(float(np.mean(errors)))

    tail = (1 - confidence) / 2
    return {
        metric: {
            "lower": float(np.quantile(scores, tail)),
            "upper": float(np.quantile(scores, 1 - tail)),
            "valid_resamples": len(scores),
        }
        for metric, scores in values.items()
    }


def run_project(
    readings_path: Path,
    targets_path: Path,
    binary_target: str | None = None,
    continuous_target: str | None = None,
    group_column: str | None = None,
    holdout_fraction: float = 0.2,
    threshold: float = 0.5,
    max_gap_minutes: float = 15.0,
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run a reproducible Week 3–5 workflow and summarize evidence for Week 6."""
    if binary_target is None and continuous_target is None:
        raise ValueError("Specify a binary target, continuous target, or both")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not readings_path.is_file():
        raise FileNotFoundError(f"Readings CSV not found: {readings_path}")
    if not targets_path.is_file():
        raise FileNotFoundError(f"Targets CSV not found: {targets_path}")

    readings = pd.read_csv(readings_path)
    targets = pd.read_csv(targets_path)
    if readings.empty:
        raise ValueError("Readings CSV contains no rows")
    if targets.empty:
        raise ValueError("Targets CSV contains no rows")

    cutoff = pd.to_datetime(readings["prediction_cutoff"], errors="coerce", utc=True)
    timestamp = pd.to_datetime(readings["timestamp"], errors="coerce", utc=True)
    if cutoff.isna().any() or timestamp.isna().any():
        raise ValueError("Readings contain invalid timestamps or prediction cutoffs")

    features = engineer_batch_features(
        readings,
        max_gap_minutes=max_gap_minutes,
    )
    report, predictions = evaluate_quality_models(
        features,
        targets,
        binary_target=binary_target,
        continuous_target=continuous_target,
        holdout_fraction=holdout_fraction,
        threshold=threshold,
        group_column=group_column,
    )

    test_batches = set(predictions["batch_id"])
    target_rows = targets.loc[targets["batch_id"].isin(test_batches)].copy()
    if group_column:
        bootstrap_groups = target_rows.set_index("batch_id")[group_column]
    else:
        bootstrap_groups = target_rows.set_index("batch_id").index.to_series()

    report["uncertainty"] = {"confidence_level": confidence, "models": {}}
    for task, task_rows in predictions.groupby("task", sort=True):
        report["uncertainty"]["models"][task] = {}
        for model_name, model_rows in task_rows.groupby("model", sort=True):
            ordered = model_rows.set_index("batch_id")
            groups = bootstrap_groups.reindex(ordered.index).to_numpy()
            if task == "classification":
                intervals = bootstrap_confidence_intervals(
                    ordered["actual"].to_numpy(),
                    ordered["predicted"].to_numpy(),
                    ordered["probability"].to_numpy(),
                    groups=groups,
                    positive_label=1,
                    samples=bootstrap_samples,
                    confidence=confidence,
                    random_state=random_state,
                )
            else:
                intervals = _regression_intervals(
                    ordered["actual"].to_numpy(dtype=float),
                    ordered["predicted"].to_numpy(dtype=float),
                    groups,
                    samples=bootstrap_samples,
                    confidence=confidence,
                    random_state=random_state,
                )
            report["uncertainty"]["models"][task][model_name] = intervals

    eligible_readings = timestamp <= cutoff
    label_profile: dict[str, Any] = {}
    if binary_target:
        labels = pd.to_numeric(targets[binary_target], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{binary_target} must contain only binary values 0 and 1")
        label_profile[binary_target] = {
            "counts": {
                str(label): int(count)
                for label, count in labels.value_counts().sort_index().items()
            },
            "positive_prevalence": float(labels.mean()),
        }
    if continuous_target:
        values = pd.to_numeric(targets[continuous_target], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(
                f"{continuous_target} must contain finite numeric values"
            )
        label_profile[continuous_target] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1))
            if values.size > 1
            else None,
            "minimum": float(values.min()),
            "median": float(values.median()),
            "maximum": float(values.max()),
        }

    report["project_run"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "inputs": {
            "readings": {
                "file": readings_path.name,
                "sha256": _sha256(readings_path),
                "rows": int(len(readings)),
                "batches": int(readings["batch_id"].nunique()),
                "sensors": int(readings["sensor"].nunique()),
                "missing_values_by_column": {
                    str(column): int(count)
                    for column, count in readings.isna().sum().items()
                    if count
                },
                "readings_after_prediction_cutoff": int((~eligible_readings).sum()),
                "timestamp_start_utc": timestamp.min().isoformat(),
                "timestamp_end_utc": timestamp.max().isoformat(),
            },
            "targets": {
                "file": targets_path.name,
                "sha256": _sha256(targets_path),
                "rows": int(len(targets)),
                "missing_values_by_column": {
                    str(column): int(count)
                    for column, count in targets.isna().sum().items()
                    if count
                },
                "label_profile": label_profile,
            },
        },
        "configuration": {
            "binary_target": binary_target,
            "continuous_target": continuous_target,
            "group_column": group_column,
            "holdout_fraction": holdout_fraction,
            "threshold": threshold if binary_target else None,
            "max_gap_minutes": max_gap_minutes,
            "bootstrap_samples": bootstrap_samples,
            "random_state": random_state,
        },
        "limitations": [
            "The latest chronological holdout is a single evaluation period, not external validation.",
            "The threshold is supplied by the user and is not optimized on validation data by this workflow.",
            "Bootstrap intervals describe sampling uncertainty for the held-out prediction set; they do not establish safety or deployment readiness.",
            "Manufacturing performance is only evidenced when real, governed production data are supplied.",
        ],
    }
    return report, features, predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end food-quality analytics workflow."
    )
    parser.add_argument("readings_csv", type=Path)
    parser.add_argument("targets_csv", type=Path)
    parser.add_argument("--binary-target")
    parser.add_argument("--continuous-target")
    parser.add_argument("--group-column")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-gap-minutes", type=float, default=15.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("week6_output"))
    args = parser.parse_args()
    if args.binary_target is None and args.continuous_target is None:
        parser.error("provide --binary-target and/or --continuous-target")

    report, features, predictions = run_project(
        args.readings_csv,
        args.targets_csv,
        binary_target=args.binary_target,
        continuous_target=args.continuous_target,
        group_column=args.group_column,
        holdout_fraction=args.holdout_fraction,
        threshold=args.threshold,
        max_gap_minutes=args.max_gap_minutes,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        random_state=args.random_state,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_dir / "batch_features.csv", index=False)
    predictions.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    report_path = args.output_dir / "project_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote batch features to {args.output_dir / 'batch_features.csv'}")
    print(f"Wrote holdout predictions to {args.output_dir / 'holdout_predictions.csv'}")
    print(f"Wrote project report to {report_path}")


if __name__ == "__main__":
    main()
