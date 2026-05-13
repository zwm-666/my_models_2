from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class PublicAcVoltageBaselineTests(unittest.TestCase):
    def test_public_metric_row_uses_test_payload_and_dataset_label(self) -> None:
        from scripts.run_public_ac_voltage_baselines import public_metric_row

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "accuracy": 1.0,
                        "macro_f1": 1.0,
                        "parameter_count": 123,
                        "test": {
                            "accuracy": 0.95,
                            "macro_f1": 0.94,
                            "weighted_f1": 0.96,
                            "inference_time_per_sample_ms": 1.25,
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = public_metric_row("6_4", "transformer", metrics_path)

        self.assertEqual(row["dataset"], "公开数据集")
        self.assertEqual(row["ratio"], "6:4")
        self.assertEqual(row["model"], "transformer")
        self.assertEqual(row["category"], "transformer")
        self.assertEqual(row["test_source"], "test")
        self.assertAlmostEqual(row["test_accuracy"], 0.95)
        self.assertAlmostEqual(row["test_macro_f1"], 0.94)
        self.assertAlmostEqual(row["test_weighted_f1"], 0.96)
        self.assertAlmostEqual(row["test_inference_ms"], 1.25)
        self.assertEqual(row["parameter_count"], 123)

    def test_public_metric_row_maps_cross_domain_protocol_labels(self) -> None:
        from scripts.run_public_ac_voltage_baselines import public_metric_row

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "accuracy": 0.8,
                        "macro_f1": 0.8,
                        "parameter_count": 456,
                        "test": {
                            "accuracy": 0.7,
                            "macro_f1": 0.6,
                            "weighted_f1": 0.65,
                            "inference_time_per_sample_ms": 2.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            old_to_new_row = public_metric_row("old_to_new", "mlp", metrics_path)
            new_to_old_row = public_metric_row("new_to_old", "mlp", metrics_path)

        self.assertEqual(old_to_new_row["ratio"], "old->new")
        self.assertEqual(new_to_old_row["ratio"], "new->old")

    def test_resolve_public_experiments_uses_single_cross_domain_run(self) -> None:
        from scripts.run_public_ac_voltage_baselines import resolve_public_experiments

        self.assertEqual(
            resolve_public_experiments("old_to_new", ["8_2", "7_3"]),
            [("old_to_new", "old_to_new")],
        )
        self.assertEqual(
            resolve_public_experiments("new_to_old", ["8_2", "7_3"]),
            [("new_to_old", "new_to_old")],
        )
        self.assertEqual(
            resolve_public_experiments("mixed_stratified", ["8_2", "7_3"]),
            [("8_2", "8_2"), ("7_3", "7_3")],
        )


if __name__ == "__main__":
    unittest.main()
