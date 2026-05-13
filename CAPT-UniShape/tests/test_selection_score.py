from __future__ import annotations

import unittest

from train import _compute_selection_score, normalize_selection_score_type


class SelectionScoreTests(unittest.TestCase):
    def test_normalize_selection_score_type(self) -> None:
        self.assertEqual(normalize_selection_score_type("macro_f1"), "macro_f1")
        self.assertEqual(normalize_selection_score_type("macro-f1-gap-penalty"), "macro_f1_gap_penalty")
        self.assertEqual(normalize_selection_score_type("joint-holdout-macro-f1"), "val_train_holdout_macro_f1")

    def test_macro_f1_gap_penalty_formula(self) -> None:
        metrics = {"macro_f1": 0.8, "train_macro_f1": 0.9}
        score = _compute_selection_score(metrics, history=[], selection_score_type="macro_f1_gap_penalty", val_metric_smoothing=3)
        self.assertAlmostEqual(score, 0.35, places=8)

    def test_macro_f1_uses_smoothing_window(self) -> None:
        history = [
            {"val_macro_f1": 0.6},
            {"val_macro_f1": 0.8},
        ]
        metrics = {"macro_f1": 0.9, "class0_recall": 0.2}
        score = _compute_selection_score(metrics, history=history, selection_score_type="macro_f1", val_metric_smoothing=3)
        self.assertAlmostEqual(score, (0.6 + 0.8 + 0.9) / 3.0, places=8)

    def test_joint_holdout_formula(self) -> None:
        metrics = {"macro_f1": 0.9, "train_holdout_macro_f1": 0.5}
        score = _compute_selection_score(metrics, history=[], selection_score_type="val_train_holdout_macro_f1", val_metric_smoothing=5)
        self.assertAlmostEqual(score, 0.78, places=8)


if __name__ == "__main__":
    unittest.main()
