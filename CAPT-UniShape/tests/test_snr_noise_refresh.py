from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path

import numpy as np

from scripts import refresh_snr_noise_results as snr


class SnrNoiseRefreshTests(unittest.TestCase):
    def test_add_snr_noise_hits_requested_level_for_targets_only(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            clean_path = tmp_path / "clean.npz"
            noisy_path = tmp_path / "snr_20dB.npz"
            x_op = np.ones((4, 2, 8), dtype=np.float32) * 2.0
            x_eis = np.ones((4, 1, 6), dtype=np.float32) * 3.0
            x_cond = np.ones((4, 3), dtype=np.float32) * 4.0
            split = np.asarray([0, 1, 2, 2], dtype=np.int64)
            labels = np.asarray([0, 1, 1, 2], dtype=np.int64)
            np.savez_compressed(clean_path, x_op=x_op, x_eis=x_eis, x_cond=x_cond, split=split, labels=labels)

            summary = snr.write_test_subset_with_snr_noise(
                clean_path,
                noisy_path,
                snr_db=20.0,
                noise_targets=["x_op", "x_eis"],
                seed=123,
            )

            with np.load(noisy_path) as noisy:
                self.assertEqual(noisy["x_op"].shape[0], 2)
                self.assertEqual(noisy["split"].tolist(), [2, 2])
                self.assertTrue(np.allclose(noisy["x_cond"], x_cond[split == 2]))
            self.assertEqual(set(summary["target_actual_snr_db"]), {"x_op", "x_eis"})
            for actual_snr in summary["target_actual_snr_db"].values():
                self.assertTrue(math.isclose(actual_snr, 20.0, abs_tol=1e-4))

    def test_merge_summary_preserves_clean_rows_and_replaces_noise(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = Path(tmpdir) / "old.csv"
            fieldnames = [
                "ratio",
                "model",
                "category",
                "snr_db",
                "test_accuracy",
                "metrics_path",
            ]
            with old_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"ratio": "8:2", "model": "mlp", "category": "deep_learning", "snr_db": "clean", "test_accuracy": "0.9", "metrics_path": "clean"})
                writer.writerow({"ratio": "8:2", "model": "mlp", "category": "deep_learning", "snr_db": "30", "test_accuracy": "0.1", "metrics_path": "old_noise"})

            merged = snr.merge_clean_rows_with_noise_rows(
                old_summary_path=old_path,
                noise_rows=[
                    {
                        "ratio": "8:2",
                        "model": "mlp",
                        "category": "deep_learning",
                        "snr_db": "40",
                        "test_accuracy": "0.8",
                        "metrics_path": "new_noise",
                    }
                ],
                model_order=["mlp"],
                snr_order=["clean", "40"],
            )

            self.assertEqual([row["snr_db"] for row in merged], ["clean", "40"])
            self.assertEqual(merged[0]["metrics_path"], "clean")
            self.assertEqual(merged[1]["metrics_path"], "new_noise")

    def test_merge_can_use_same_run_clean_rows_instead_of_old_summary(self) -> None:
        merged = snr.merge_clean_rows_with_noise_rows(
            clean_rows_by_model={
                "mlp": {
                    "ratio": "8:2",
                    "model": "mlp",
                    "category": "deep_learning",
                    "snr_db": "clean",
                    "test_accuracy": "1.0",
                    "metrics_path": "same_run_clean",
                }
            },
            noise_rows=[
                {
                    "ratio": "8:2",
                    "model": "mlp",
                    "category": "deep_learning",
                    "snr_db": "40",
                    "test_accuracy": "1.0",
                    "metrics_path": "new_noise",
                }
            ],
            model_order=["mlp"],
            snr_order=["clean", "40"],
        )

        self.assertEqual(merged[0]["metrics_path"], "same_run_clean")
        self.assertEqual(float(merged[0]["test_accuracy"]), 1.0)

    def test_cli_defaults_to_same_run_clean_source(self) -> None:
        args = snr.parse_args([])
        self.assertEqual(args.clean_source, "same-run")

    def test_compact_summary_uses_required_headers_and_four_decimals(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "summary.csv"
            snr.write_compact_summary(
                out_path,
                [
                    {
                        "model": "mlp",
                        "snr_db": "40",
                        "test_accuracy": 0.987654,
                        "accuracy_drop": -0.012345,
                        "test_macro_f1": 0.876543,
                        "macro_f1_drop": 0.0,
                        "test_weighted_f1": 0.765432,
                        "weighted_f1_drop": 0.111111,
                        "data_path": "data.npz",
                        "metrics_path": "metrics.json",
                        "clean_alignment_source": "best.ckpt",
                    }
                ],
            )

            with out_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, snr.COMPACT_SUMMARY_FIELDNAMES)
                rows = list(reader)

            self.assertEqual(rows[0]["accuracy"], "0.9877")
            self.assertEqual(rows[0]["accuracy_drop"], "-0.0123")
            self.assertEqual(rows[0]["macro_f1"], "0.8765")
            self.assertEqual(rows[0]["weighted_f1_drop"], "0.1111")

    def test_row_from_metrics_does_not_clip_noisy_better_than_clean(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(
                """{
  "test": {
    "accuracy": 0.95,
    "macro_f1": 0.94,
    "weighted_f1": 0.93,
    "inference_time_per_sample_ms": 1.0,
    "per_class_f1": []
  },
  "parameter_count": 1
}""",
                encoding="utf-8",
            )
            row = snr._row_from_metrics(
                model_key="mlp",
                snr_db=40.0,
                actual_snr_db_mean=40.0,
                noise_targets=["x_op"],
                data_path=Path("snr_40dB.npz"),
                metrics_path=metrics_path,
                clean_row={"test_accuracy": 0.90, "test_macro_f1": 0.89, "test_weighted_f1": 0.88},
                clean_alignment_source="same_checkpoint",
            )

            self.assertAlmostEqual(row["accuracy_drop"], -0.05)
            self.assertAlmostEqual(row["macro_f1_drop"], -0.05)
            self.assertAlmostEqual(row["weighted_f1_drop"], -0.05)

    def test_merge_can_update_selected_models_while_preserving_others(self) -> None:
        existing_rows = [
            {
                "ratio": "8:2",
                "model": "mlp",
                "category": "deep_learning",
                "snr_db": "clean",
                "test_accuracy": "0.90",
                "metrics_path": "mlp_clean",
            },
            {
                "ratio": "8:2",
                "model": "proposed",
                "category": "proposed",
                "snr_db": "clean",
                "test_accuracy": "1.00",
                "metrics_path": "proposed_clean_old",
            },
            {
                "ratio": "8:2",
                "model": "mlp",
                "category": "deep_learning",
                "snr_db": "40",
                "test_accuracy": "0.80",
                "metrics_path": "mlp_40_old",
            },
            {
                "ratio": "8:2",
                "model": "proposed",
                "category": "proposed",
                "snr_db": "40",
                "test_accuracy": "0.70",
                "metrics_path": "proposed_40_old",
            },
        ]

        merged = snr.merge_updated_rows_with_existing_rows(
            existing_rows=existing_rows,
            updated_rows=[
                {
                    "ratio": "8:2",
                    "model": "proposed",
                    "category": "proposed",
                    "snr_db": "clean",
                    "test_accuracy": "1.00",
                    "metrics_path": "proposed_clean_old",
                },
                {
                    "ratio": "8:2",
                    "model": "proposed",
                    "category": "proposed",
                    "snr_db": "40",
                    "test_accuracy": "0.88",
                    "metrics_path": "proposed_40_new",
                },
            ],
            model_order=["mlp", "proposed"],
            snr_order=["clean", "40"],
        )

        self.assertEqual(
            [(row["model"], row["snr_db"], row["metrics_path"]) for row in merged],
            [
                ("mlp", "clean", "mlp_clean"),
                ("proposed", "clean", "proposed_clean_old"),
                ("mlp", "40", "mlp_40_old"),
                ("proposed", "40", "proposed_40_new"),
            ],
        )

    def test_parse_model_path_overrides_and_resolve_reuse_artifacts(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "mlp_legacy"
            model_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = model_dir / "metrics.json"
            checkpoint_path = model_dir / "best.ckpt"
            metrics_path.write_text("{}", encoding="utf-8")
            checkpoint_path.write_bytes(b"checkpoint")

            overrides = snr.parse_model_path_overrides([f"mlp={model_dir}"])
            self.assertEqual(overrides["mlp"], model_dir)

            resolved_metrics, resolved_checkpoint = snr.resolve_reuse_clean_artifacts("mlp", model_dir)
            self.assertEqual(resolved_metrics, metrics_path)
            self.assertEqual(resolved_checkpoint, checkpoint_path)

    def test_noise_seed_matches_cli_example(self) -> None:
        self.assertEqual(snr.noise_seed_for_snr(44, 20.0), 44020000)
        self.assertEqual(snr.noise_seed_for_snr(44, 35.0), 44035000)

    def test_load_reference_clean_rows_from_test_summary(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metrics_path = tmp / "transformer_metrics.json"
            metrics_path.write_text(
                """{
  "test": {
    "accuracy": 0.9454545454545454,
    "macro_f1": 0.9328756674294431,
    "weighted_f1": 0.9485749947992511,
    "inference_time_per_sample_ms": 0.5315909090910509,
    "per_class_f1": [{"class_id": 0, "precision": 1.0, "recall": 0.875, "f1": 0.9333333333333333}]
  },
  "parameter_count": 109763
}""",
                encoding="utf-8",
            )
            summary_path = tmp / "test_summary.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "ratio",
                        "model",
                        "category",
                        "test_accuracy",
                        "test_macro_f1",
                        "test_weighted_f1",
                        "test_inference_ms",
                        "parameter_count",
                        "metrics_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "ratio": "8:2",
                        "model": "transformer",
                        "category": "transformer",
                        "test_accuracy": "0.9454545454545454",
                        "test_macro_f1": "0.9328756674294431",
                        "test_weighted_f1": "0.9485749947992511",
                        "test_inference_ms": "0.5315909090910509",
                        "parameter_count": "109763",
                        "metrics_path": str(metrics_path),
                    }
                )

            rows = snr.load_reference_clean_rows(summary_path, ["transformer"], tmp / "clean.npz")
            self.assertEqual(rows["transformer"]["snr_db"], "clean")
            self.assertEqual(rows["transformer"]["metrics_path"], str(metrics_path))
            self.assertAlmostEqual(float(rows["transformer"]["test_accuracy"]), 0.9454545454545454)

    def test_workbook_rows_format_display_names_and_percentages(self) -> None:
        row = {
            "model": "proposed",
            "snr_db": "40",
            "test_accuracy": 0.875,
            "accuracy_drop": 0.125,
            "test_macro_f1": 0.75,
            "macro_f1_drop": 0.25,
            "test_weighted_f1": 0.8,
            "class0_recall": 0.5,
            "test_inference_ms": 12.3456,
        }

        self.assertEqual(
            snr.summary_row_to_workbook_values(row),
            [
                "所提模型",
                "40",
                "87.50%",
                "12.50%",
                "75.00%",
                "25.00%",
                "80.00%",
                "50.00%",
                "12.346",
            ],
        )

    def test_normalize_legacy_summary_row_schema(self) -> None:
        row = snr.normalize_summary_row_schema(
            {
                "model": "proposed",
                "snr_db": "clean",
                "accuracy": "1.0",
                "macro_f1": "1.0",
                "weighted_f1": "1.0",
                "data_path": "clean.npz",
                "metrics_path": "metrics.json",
            }
        )

        self.assertEqual(row["model"], "proposed")
        self.assertEqual(row["category"], "proposed")
        self.assertEqual(row["test_accuracy"], "1.0")
        self.assertEqual(row["test_macro_f1"], "1.0")
        self.assertEqual(row["test_weighted_f1"], "1.0")


if __name__ == "__main__":
    unittest.main()
