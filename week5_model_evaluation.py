from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.validation import check_consistent_length


RANDOM_STATE = 42


def _as_binary(values: Any, positive_label: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return array == positive_label


def _probabilities(values: Any, expected_size: int) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or len(scores) != expected_size:
        raise ValueError("Probabilities must be one-dimensional and match the labels")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    return scores


def _expected_calibration_error(
    actual: np.ndarray,
    probability: np.ndarray,
    bins: int,
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total_error = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if selected.any():
            total_error += selected.mean() * abs(
                probability[selected].mean() - actual[selected].mean()
            )
    return float(total_error)


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_probability: Any | None = None,
    positive_label: Any = 1,
    calibration_bins: int = 10,
) -> dict[str, int | float | None]:
    """Calculate binary classification and probability-quality metrics."""
    actual = _as_binary(y_true, positive_label, "y_true")
    predicted = _as_binary(y_pred, positive_label, "y_pred")
    check_consistent_length(actual, predicted)
    labels = pd.unique(np.concatenate((np.asarray(y_true), np.asarray(y_pred))))
    if len(labels) > 2:
        raise ValueError("Classification metrics require a binary target")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be at least 1")

    true_positive = int(np.sum(actual & predicted))
    false_positive = int(np.sum(~actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    true_negative = int(np.sum(~actual & ~predicted))
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / positive_count if positive_count else None
    specificity = true_negative / negative_count if negative_count else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else 0.0 if recall is not None else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )

    result: dict[str, int | float | None] = {
        "n": int(len(actual)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "prevalence": float(actual.mean()),
        "accuracy": float(np.mean(actual == predicted)),
        "precision": float(precision),
        "recall": float(recall) if recall is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "f1": float(f1) if f1 is not None else None,
        "balanced_accuracy": (
            float(balanced_accuracy) if balanced_accuracy is not None else None
        ),
    }

    if y_probability is not None:
        probability = _probabilities(y_probability, len(actual))
        result["brier_score"] = float(np.mean((probability - actual) ** 2))
        result["expected_calibration_error"] = _expected_calibration_error(
            actual, probability, calibration_bins
        )
        result["roc_auc"] = (
            float(roc_auc_score(actual, probability))
            if np.unique(actual).size == 2
            else None
        )
        result["pr_auc"] = (
            float(average_precision_score(actual, probability))
            if positive_count
            else None
        )
    return result


def bootstrap_confidence_intervals(
    y_true: Any,
    y_pred: Any,
    y_probability: Any | None = None,
    groups: Any | None = None,
    positive_label: Any = 1,
    samples: int = 1000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float | int]]:
    """Estimate percentile intervals by resampling rows or whole groups."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    check_consistent_length(actual, predicted)
    probability = (
        _probabilities(y_probability, len(actual))
        if y_probability is not None
        else None
    )
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    if groups is None:
        units = np.arange(len(actual))
        row_indices = [np.array([index]) for index in units]
    else:
        group_values = np.asarray(groups)
        check_consistent_length(actual, group_values)
        if pd.isna(group_values).any():
            raise ValueError("groups cannot contain missing values")
        units = pd.unique(group_values)
        row_indices = [np.flatnonzero(group_values == unit) for unit in units]
    if len(units) < 2:
        raise ValueError("At least two bootstrap units are required")

    rng = np.random.default_rng(random_state)
    collected: dict[str, list[float]] = {}
    for _ in range(samples):
        chosen = rng.integers(0, len(units), size=len(units))
        indices = np.concatenate([row_indices[index] for index in chosen])
        scores = classification_metrics(
            actual[indices],
            predicted[indices],
            probability[indices] if probability is not None else None,
            positive_label=positive_label,
        )
        for name, value in scores.items():
            if isinstance(value, (float, int)) and np.isfinite(value):
                collected.setdefault(name, []).append(float(value))

    tail = (1 - confidence) / 2
    intervals = {}
    for name, values in collected.items():
        if values:
            lower, upper = np.quantile(values, [tail, 1 - tail])
            intervals[name] = {
                "lower": float(lower),
                "upper": float(upper),
                "valid_resamples": len(values),
            }
    return intervals


def chronological_split_indices(
    timestamps: Any,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by timestamp while keeping tied timestamps in the same partition."""
    dates = pd.to_datetime(pd.Series(timestamps), errors="coerce", utc=True)
    if dates.empty or dates.isna().any():
        raise ValueError("timestamps must be non-empty and valid")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    unique_dates = dates.drop_duplicates().sort_values().to_numpy()
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct timestamps are required")
    split_at = int(np.floor(len(unique_dates) * (1 - test_fraction)))
    split_at = min(max(split_at, 1), len(unique_dates) - 1)
    boundary = unique_dates[split_at]
    train = np.flatnonzero(dates.to_numpy() < boundary)
    test = np.flatnonzero(dates.to_numpy() >= boundary)
    return train, test


def evaluate_cross_validation(
    estimator: Any,
    features: Any,
    y_true: Any,
    n_splits: int = 5,
    groups: Any | None = None,
    positive_label: Any = 1,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Return out-of-fold and per-fold metrics from stratified validation."""
    target = np.asarray(y_true)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if np.unique(target).size != 2:
        raise ValueError("Cross-validation requires exactly two target classes")

    if groups is None:
        if np.unique(target, return_counts=True)[1].min() < n_splits:
            raise ValueError("Each target class must have at least n_splits rows")
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = splitter.split(np.zeros(len(target)), target)
    else:
        group_values = np.asarray(groups)
        check_consistent_length(target, group_values)
        if pd.isna(group_values).any():
            raise ValueError("groups cannot contain missing values")
        if pd.unique(group_values).size < n_splits:
            raise ValueError("There must be at least n_splits distinct groups")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = splitter.split(np.zeros(len(target)), target, group_values)

    prediction = np.empty(len(target), dtype=target.dtype)
    probability = np.full(len(target), np.nan, dtype=float)
    visits = np.zeros(len(target), dtype=int)
    fold_metrics = []
    for train_indices, test_indices in splits:
        model = clone(estimator)
        model.fit(_take_rows(features, train_indices), target[train_indices])
        classes = np.asarray(model.classes_)
        positive_indices = np.flatnonzero(classes == positive_label)
        if positive_indices.size != 1:
            raise ValueError(
                "A training fold does not contain the requested positive class"
            )
        fold_probability = np.asarray(
            model.predict_proba(_take_rows(features, test_indices)), dtype=float
        )[:, positive_indices[0]]
        fold_prediction = model.predict(_take_rows(features, test_indices))
        prediction[test_indices] = fold_prediction
        probability[test_indices] = fold_probability
        visits[test_indices] += 1
        fold_metrics.append(
            classification_metrics(
                target[test_indices],
                fold_prediction,
                fold_probability,
                positive_label=positive_label,
            )
        )

    if not np.all(visits == 1):
        raise RuntimeError("Cross-validation did not predict every row exactly once")
    return {
        "out_of_fold": classification_metrics(
            target, prediction, probability, positive_label=positive_label
        ),
        "folds": fold_metrics,
    }


def _take_rows(values: Any, indices: np.ndarray) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return np.asarray(values)[indices]


def population_stability_index(
    reference: Any,
    current: Any,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Measure numeric feature distribution shift using reference quantile bins."""
    baseline = np.asarray(reference, dtype=float)
    recent = np.asarray(current, dtype=float)
    if baseline.ndim != 1 or recent.ndim != 1 or not baseline.size or not recent.size:
        raise ValueError("reference and current must be non-empty one-dimensional arrays")
    if not np.isfinite(baseline).all() or not np.isfinite(recent).all():
        raise ValueError("Drift inputs must contain only finite values")
    if bins < 2 or not 0 < epsilon < 1:
        raise ValueError("bins must be at least 2 and epsilon must be between 0 and 1")

    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(baseline, quantiles))
    if np.ptp(baseline) == 0:
        width = max(abs(float(baseline[0])) * 1e-6, 1e-6)
        edges = np.array([baseline[0] - width, baseline[0] + width])
    reference_counts = np.histogram(baseline, bins=np.r_[-np.inf, edges, np.inf])[0]
    current_counts = np.histogram(recent, bins=np.r_[-np.inf, edges, np.inf])[0]
    reference_share = np.maximum(reference_counts / len(baseline), epsilon)
    current_share = np.maximum(current_counts / len(recent), epsilon)
    return float(np.sum((current_share - reference_share) * np.log(
        current_share / reference_share
    )))


def _parse_label(value: str, observed: pd.Series) -> Any:
    if value in set(observed.dropna().tolist()):
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if pd.api.types.is_integer_dtype(observed.dtype) and number.is_integer():
        return int(number)
    return number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved binary classification predictions."
    )
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--actual-column", default="actual")
    parser.add_argument("--predicted-column", default="predicted")
    parser.add_argument("--probability-column", default="probability")
    parser.add_argument("--positive-label", default="1")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--group-column")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.predictions_csv)
    required = {args.actual_column}
    if args.threshold is None:
        required.add(args.predicted_column)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    positive_label = _parse_label(args.positive_label, data[args.actual_column])
    probability = None
    predictions = data[args.predicted_column] if args.threshold is None else None
    if args.probability_column in data.columns:
        probability = _probabilities(
            data[args.probability_column], len(data)
        )
    if args.threshold is not None:
        if not 0 <= args.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if probability is None:
            raise ValueError(
                f"Threshold evaluation requires {args.probability_column!r}"
            )
        predictions = np.where(
            probability >= args.threshold,
            positive_label,
            _negative_label(data[args.actual_column], positive_label),
        )

    scores = classification_metrics(
        data[args.actual_column],
        predictions,
        probability,
        positive_label=positive_label,
    )
    groups = data[args.group_column] if args.group_column else None
    intervals = bootstrap_confidence_intervals(
        data[args.actual_column],
        predictions,
        probability,
        groups=groups,
        positive_label=positive_label,
        samples=args.bootstrap_samples,
    )
    result = {
        "positive_label": positive_label,
        "decision_threshold": args.threshold,
        "metrics": scores,
        "confidence_intervals": intervals,
        "bootstrap_unit": args.group_column or "row",
    }
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _negative_label(values: pd.Series, positive_label: Any) -> Any:
    labels = [label for label in pd.unique(values.dropna()) if label != positive_label]
    if len(labels) != 1:
        raise ValueError("Actual labels must contain exactly one negative class")
    return labels[0]


if __name__ == "__main__":
    main()
