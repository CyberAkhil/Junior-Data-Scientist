import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from week3_feature_engineering import (
    add_interaction,
    add_polynomial_features,
    cluster_batch_profiles,
    engineer_batch_features,
    evaluate_regression,
    main,
)


def make_readings() -> pd.DataFrame:
    rows = []
    for batch_number in range(5):
        start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=batch_number)
        batch_id = f"batch-{batch_number}"
        cutoff = start + pd.Timedelta(minutes=10)
        values = [10 + batch_number, 14 + batch_number]
        for minute, value in zip((0, 5), values):
            rows.append(
                {
                    "batch_id": batch_id,
                    "timestamp": start + pd.Timedelta(minutes=minute),
                    "prediction_cutoff": cutoff,
                    "sensor": "temperature",
                    "value": value,
                    "target": 11 + batch_number,
                    "lower_limit": 9 + batch_number,
                    "upper_limit": 12 + batch_number,
                }
            )
        rows.append(
            {
                "batch_id": batch_id,
                "timestamp": cutoff + pd.Timedelta(minutes=1),
                "prediction_cutoff": cutoff,
                "sensor": "temperature",
                "value": 1000,
                "target": 11 + batch_number,
                "lower_limit": 9 + batch_number,
                "upper_limit": 12 + batch_number,
            }
        )
    return pd.DataFrame(rows)


class FeatureEngineeringTests(unittest.TestCase):
    def test_features_respect_cutoff_and_measure_spec_exposure(self):
        features = engineer_batch_features(make_readings(), max_gap_minutes=6)
        first = features.set_index("batch_id").loc["batch-0"]

        self.assertEqual(first["temperature__reading_count"], 2)
        self.assertEqual(first["temperature__mean"], 12)
        self.assertEqual(first["temperature__mean_target_deviation"], 1)
        self.assertEqual(first["temperature__slope_per_minute"], 0.8)
        self.assertEqual(first["temperature__within_spec_fraction"], 0.5)
        self.assertEqual(first["temperature__observed_minutes"], 5)
        self.assertEqual(first["temperature__in_spec_minutes"], 5)
        self.assertEqual(first["temperature__out_of_spec_minutes"], 0)

    def test_duplicate_events_are_rejected(self):
        readings = make_readings()
        readings = pd.concat([readings, readings.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            engineer_batch_features(readings)

    def test_polynomial_and_selected_interaction_features(self):
        features = pd.DataFrame(
            {"temperature__mean": [1.0, 2.0], "flow__mean": [3.0, 4.0]}
        )
        expanded = add_polynomial_features(features, ["temperature__mean"])
        interacted = add_interaction(expanded, "temperature__mean", "flow__mean")

        self.assertIn("poly__temperature__mean^2", interacted.columns)
        self.assertEqual(
            interacted["temperature__mean__x__flow__mean"].tolist(),
            [3.0, 8.0],
        )

    def test_chronological_regression_and_clustering(self):
        features = engineer_batch_features(make_readings(), max_gap_minutes=6)
        targets = pd.DataFrame(
            {
                "batch_id": [f"batch-{i}" for i in range(5)],
                "quality_target": [20.0, 19.0, 18.0, 17.0, 16.0],
            }
        )
        scores = evaluate_regression(features, targets)
        clusters = cluster_batch_profiles(features, n_clusters=2)

        self.assertEqual(
            set(scores["model"]),
            {"median_baseline", "ridge", "decision_tree"},
        )
        self.assertEqual(len(clusters), 5)
        self.assertEqual(clusters["cluster"].nunique(), 2)

    def test_command_line_writes_features_and_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readings_path = root / "readings.csv"
            targets_path = root / "targets.csv"
            output_path = root / "features.csv"
            make_readings().to_csv(readings_path, index=False)
            pd.DataFrame(
                {
                    "batch_id": [f"batch-{i}" for i in range(5)],
                    "quality_target": [20.0, 19.0, 18.0, 17.0, 16.0],
                }
            ).to_csv(targets_path, index=False)
            arguments = [
                "week3_feature_engineering.py",
                str(readings_path),
                "--output",
                str(output_path),
                "--targets-csv",
                str(targets_path),
                "--clusters",
                "2",
                "--interaction",
                "temperature__mean",
                "temperature__std",
                "--polynomial-features",
                "temperature__mean_target_deviation",
            ]

            with (
                patch("sys.argv", arguments),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                main()

            saved = pd.read_csv(output_path)
            self.assertEqual(len(saved), 5)
            self.assertIn("cluster", saved.columns)
            self.assertIn("temperature__mean__x__temperature__std", saved.columns)
            self.assertIn("ridge", output.getvalue())


if __name__ == "__main__":
    unittest.main()
