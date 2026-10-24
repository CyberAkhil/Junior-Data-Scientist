import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from week3_feature_engineering import engineer_batch_features
from week4_predictive_modeling import evaluate_quality_models, main


def make_data(batch_count=30):
    rows = []
    targets = []
    for number in range(batch_count):
        start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=number)
        cutoff = start + pd.Timedelta(minutes=10)
        batch_id = f"batch-{number}"
        value = float(number)
        for minute, reading in ((0, value), (5, value + 1)):
            rows.append(
                {
                    "batch_id": batch_id,
                    "timestamp": start + pd.Timedelta(minutes=minute),
                    "prediction_cutoff": cutoff,
                    "sensor": "temperature",
                    "value": reading,
                }
            )
        targets.append(
            {
                "batch_id": batch_id,
                "nonconforming": int(number % 3 == 0),
                "quality_result": 20 + value * 0.5,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(targets)


class PredictiveModelingTests(unittest.TestCase):
    def test_models_report_temporal_classification_and_regression(self):
        readings, targets = make_data()
        targets["future_lab_result"] = targets["quality_result"] * 10
        targets["campaign"] = [number // 6 for number in range(len(targets))]
        features = engineer_batch_features(readings)
        report, predictions = evaluate_quality_models(
            features,
            targets,
            binary_target="nonconforming",
            continuous_target="quality_result",
            group_column="campaign",
        )

        self.assertEqual(report["train_batches"], 24)
        self.assertEqual(report["test_batches"], 6)
        self.assertNotIn("future_lab_result", report["features"])
        self.assertNotIn("campaign", report["features"])
        self.assertEqual(
            set(report["classification"]["models"]),
            {
                "prevalence_baseline",
                "logistic_regression",
                "decision_tree",
                "random_forest",
            },
        )
        self.assertEqual(
            set(report["regression"]["models"]),
            {"median_baseline", "ridge", "decision_tree", "random_forest"},
        )
        self.assertEqual(len(predictions), 48)
        self.assertEqual(
            predictions.groupby("task")["batch_id"].nunique().to_dict(),
            {"classification": 6, "regression": 6},
        )

    def test_group_split_rejects_groups_crossing_holdout_boundary(self):
        readings, targets = make_data()
        features = engineer_batch_features(readings)
        targets["campaign"] = ["campaign-a"] * 15 + ["campaign-b"] * 15

        with self.assertRaisesRegex(ValueError, "cross the chronological boundary"):
            evaluate_quality_models(
                features,
                targets,
                continuous_target="quality_result",
                group_column="campaign",
            )

    def test_cli_writes_predictions_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            readings, targets = make_data()
            readings_path = root / "readings.csv"
            targets_path = root / "targets.csv"
            predictions_path = root / "predictions.csv"
            report_path = root / "report.json"
            readings.to_csv(readings_path, index=False)
            targets.to_csv(targets_path, index=False)
            arguments = [
                "week4_predictive_modeling.py",
                str(readings_path),
                str(targets_path),
                "--binary-target",
                "nonconforming",
                "--continuous-target",
                "quality_result",
                "--predictions-output",
                str(predictions_path),
                "--report-output",
                str(report_path),
            ]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(
                io.StringIO()
            ):
                main()

            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            saved_predictions = pd.read_csv(predictions_path)
            self.assertIn("logistic_regression", saved_report["classification"]["models"])
            self.assertEqual(len(saved_predictions), 48)
            self.assertEqual(set(saved_predictions["task"]), {"classification", "regression"})


if __name__ == "__main__":
    unittest.main()
