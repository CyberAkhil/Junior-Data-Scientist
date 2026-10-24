import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from week5_model_evaluation import (
    bootstrap_confidence_intervals,
    chronological_split_indices,
    classification_metrics,
    evaluate_cross_validation,
    main,
    population_stability_index,
)


class ModelEvaluationTests(unittest.TestCase):
    def test_binary_metrics_and_probability_scores(self):
        scores = classification_metrics(
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [0.1, 0.7, 0.8, 0.4],
        )

        self.assertEqual(scores["true_positive"], 1)
        self.assertEqual(scores["false_positive"], 1)
        self.assertEqual(scores["false_negative"], 1)
        self.assertEqual(scores["true_negative"], 1)
        self.assertEqual(scores["accuracy"], 0.5)
        self.assertEqual(scores["recall"], 0.5)
        self.assertAlmostEqual(scores["brier_score"], 0.225)
        self.assertIn("pr_auc", scores)

    def test_metrics_reject_multiclass_labels(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            classification_metrics([0, 1, 2], [0, 1, 2])

    def test_group_bootstrap_uses_whole_groups(self):
        intervals = bootstrap_confidence_intervals(
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1],
            groups=["a", "a", "b", "b", "c", "c"],
            samples=40,
            random_state=4,
        )

        self.assertEqual(intervals["n"]["lower"], 6.0)
        self.assertEqual(intervals["n"]["upper"], 6.0)
        self.assertEqual(intervals["n"]["valid_resamples"], 40)

    def test_chronological_split_keeps_equal_timestamps_together(self):
        dates = pd.to_datetime(
            ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"]
        )
        train, test = chronological_split_indices(dates, test_fraction=0.25)

        self.assertEqual(train.tolist(), [0, 1, 2])
        self.assertEqual(test.tolist(), [3])

    def test_cross_validation_reports_out_of_fold_predictions(self):
        features = pd.DataFrame(
            {"signal": np.arange(60), "noise": np.tile([0, 1], 30)}
        )
        target = (features["signal"] >= 30).astype(int).to_numpy()
        groups = np.repeat(np.arange(30), 2)
        result = evaluate_cross_validation(
            LogisticRegression(),
            features,
            target,
            n_splits=3,
            groups=groups,
        )

        self.assertEqual(len(result["folds"]), 3)
        self.assertEqual(result["out_of_fold"]["n"], 60)
        self.assertGreaterEqual(result["out_of_fold"]["recall"], 0.9)

    def test_population_stability_index_detects_distribution_shift(self):
        reference = np.linspace(0, 1, 1000)
        unchanged = population_stability_index(reference, reference, bins=10)
        shifted = population_stability_index(reference, reference + 2, bins=10)

        self.assertAlmostEqual(unchanged, 0.0)
        self.assertGreater(shifted, unchanged)

    def test_population_stability_index_detects_shift_from_constant_reference(self):
        score = population_stability_index([5.0] * 30, [6.0] * 30)

        self.assertGreater(score, 0.0)

    def test_cli_writes_json_metrics_and_group_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.csv"
            output = root / "metrics.json"
            pd.DataFrame(
                {
                    "actual": [0, 1, 0, 1],
                    "predicted": [0, 1, 0, 0],
                    "probability": [0.1, 0.8, 0.3, 0.4],
                    "batch_id": ["a", "b", "c", "d"],
                }
            ).to_csv(source, index=False)
            arguments = [
                "week5_model_evaluation.py",
                str(source),
                "--group-column",
                "batch_id",
                "--bootstrap-samples",
                "20",
                "--output",
                str(output),
            ]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(
                io.StringIO()
            ):
                main()

            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["metrics"]["n"], 4)
            self.assertEqual(result["bootstrap_unit"], "batch_id")

    def test_cli_applies_validation_threshold_to_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "predictions.csv"
            pd.DataFrame(
                {
                    "actual": [0, 1, 0, 1],
                    "probability": [0.2, 0.7, 0.6, 0.9],
                }
            ).to_csv(source, index=False)
            arguments = [
                "week5_model_evaluation.py",
                str(source),
                "--threshold",
                "0.65",
                "--bootstrap-samples",
                "10",
            ]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(
                io.StringIO()
            ) as output:
                main()

            result = json.loads(output.getvalue())
            self.assertEqual(result["metrics"]["true_positive"], 2)
            self.assertEqual(result["metrics"]["false_positive"], 0)


if __name__ == "__main__":
    unittest.main()
