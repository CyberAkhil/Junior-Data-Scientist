import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from week6_project_workflow import main, run_project


def make_inputs(batch_count=30):
    readings = []
    targets = []
    for number in range(batch_count):
        start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=number)
        cutoff = start + pd.Timedelta(minutes=10)
        batch_id = f"batch-{number}"
        value = float(number)
        for minute, reading in ((0, value), (5, value + 1)):
            readings.append(
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
                "campaign": number // 2,
            }
        )
    readings.append(
        {
            "batch_id": "batch-0",
            "timestamp": pd.Timestamp("2025-01-01 00:15"),
            "prediction_cutoff": pd.Timestamp("2025-01-01 00:10"),
            "sensor": "temperature",
            "value": 999,
        }
    )
    return pd.DataFrame(readings), pd.DataFrame(targets)


class ProjectWorkflowTests(unittest.TestCase):
    def test_run_summarizes_profile_models_and_grouped_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            readings, targets = make_inputs()
            readings_path = root / "readings.csv"
            targets_path = root / "targets.csv"
            readings.to_csv(readings_path, index=False)
            targets.to_csv(targets_path, index=False)

            report, features, predictions = run_project(
                readings_path,
                targets_path,
                binary_target="nonconforming",
                continuous_target="quality_result",
                group_column="campaign",
                bootstrap_samples=25,
            )

            run = report["project_run"]
            self.assertEqual(run["inputs"]["readings"]["batches"], 30)
            self.assertEqual(
                run["inputs"]["readings"]["readings_after_prediction_cutoff"], 1
            )
            self.assertEqual(
                run["inputs"]["targets"]["label_profile"]["nonconforming"][
                    "positive_prevalence"
                ],
                10 / 30,
            )
            self.assertEqual(len(features), 30)
            self.assertEqual(report["test_batches"], 6)
            self.assertEqual(len(predictions), 48)
            self.assertIn(
                "logistic_regression",
                report["uncertainty"]["models"]["classification"],
            )
            self.assertIn(
                "mae",
                report["uncertainty"]["models"]["regression"]["ridge"],
            )

    def test_cli_writes_features_predictions_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            readings, targets = make_inputs()
            readings_path = root / "readings.csv"
            targets_path = root / "targets.csv"
            output_dir = root / "results"
            readings.to_csv(readings_path, index=False)
            targets.to_csv(targets_path, index=False)
            arguments = [
                "week6_project_workflow.py",
                str(readings_path),
                str(targets_path),
                "--binary-target",
                "nonconforming",
                "--continuous-target",
                "quality_result",
                "--group-column",
                "campaign",
                "--bootstrap-samples",
                "20",
                "--output-dir",
                str(output_dir),
            ]

            with patch("sys.argv", arguments), contextlib.redirect_stdout(
                io.StringIO()
            ):
                main()

            report = json.loads(
                (output_dir / "project_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["train_batches"], 24)
            self.assertTrue((output_dir / "batch_features.csv").is_file())
            self.assertTrue((output_dir / "holdout_predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
