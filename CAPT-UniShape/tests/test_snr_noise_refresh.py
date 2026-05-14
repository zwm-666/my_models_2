from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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
        self.assertEqual(args.clean_source, "reference")
        self.assertEqual(args.summary_path, "results\\噪声对齐论文实验新表.csv")

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
                "12.346",
            ],
        )

    def test_summary_fieldnames_do_not_include_class0_columns(self) -> None:
        self.assertNotIn("class0_precision", snr.SUMMARY_FIELDNAMES)
        self.assertNotIn("class0_recall", snr.SUMMARY_FIELDNAMES)
        self.assertNotIn("class0_f1", snr.SUMMARY_FIELDNAMES)
        self.assertEqual(
            snr.EXPORT_FIELDNAMES,
            [
                "model",
                "snr_db",
                "test_accuracy",
                "accuracy_drop",
                "test_macro_f1",
                "macro_f1_drop",
                "test_weighted_f1",
                "weighted_f1_drop",
                "test_inference_ms",
                "parameter_count",
                "data_path",
                "metrics_path",
                "alignment_source",
            ],
        )

    def test_update_workbook_snr_sheet_uses_times_new_roman_font(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "snr.xlsx"
            workbook = Workbook()
            workbook.active.title = "Sheet1"
            workbook.save(workbook_path)
            workbook.close()

            snr.update_workbook_snr_sheet(
                workbook_path,
                [
                    {
                        "model": "proposed",
                        "snr_db": "40",
                        "test_accuracy": 1.0,
                        "accuracy_drop": 0.0,
                        "test_macro_f1": 1.0,
                        "macro_f1_drop": 0.0,
                        "test_weighted_f1": 1.0,
                        "test_inference_ms": 20.1142,
                    }
                ],
            )

            saved = load_workbook(workbook_path, read_only=True)
            try:
                worksheet = saved["SNR噪声对比"]
                self.assertEqual(worksheet["A1"].font.name, "Times New Roman")
                self.assertEqual(worksheet["A2"].font.name, "Times New Roman")
                self.assertEqual(worksheet["H2"].font.name, "Times New Roman")
            finally:
                saved.close()
                del saved
                import gc

                gc.collect()

    def test_export_row_uses_percent_strings_with_two_decimals(self) -> None:
        row = snr.export_summary_row(
            {
                "model": "proposed",
                "snr_db": "40",
                "test_accuracy": 1.0,
                "accuracy_drop": 0.12345,
                "test_macro_f1": 0.815384615,
                "macro_f1_drop": 0.184615384,
                "test_weighted_f1": 0.931934732,
                "weighted_f1_drop": 0.068065268,
                "test_inference_ms": 25.2239,
                "parameter_count": 6496573,
                "data_path": "clean.npz",
                "metrics_path": "metrics.json",
                "clean_alignment_source": "comparison_summary_reference",
            }
        )
        self.assertEqual(row["test_accuracy"], "100.00%")
        self.assertEqual(row["accuracy_drop"], "12.35%")
        self.assertEqual(row["test_macro_f1"], "81.54%")
        self.assertEqual(row["macro_f1_drop"], "18.46%")
        self.assertEqual(row["test_weighted_f1"], "93.19%")
        self.assertEqual(row["weighted_f1_drop"], "6.81%")
        self.assertEqual(row["test_inference_ms"], "25.2239")
        self.assertEqual(row["alignment_source"], "comparison_summary_reference")

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

    def test_load_model_order_from_current_comparison_config_excludes_models(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "current_comparison_models.yaml"
            config_path.write_text(
                """proposed:
  model_name: proposed_model
baselines:
  logreg: {}
  svm: {}
  random_forest: {}
  mlp: {}
  cnn1d: {}
  transformer: {}
  itransformer: {}
excluded_models:
  - svm
  - lstm
""",
                encoding="utf-8",
            )

            self.assertEqual(
                snr.load_model_order_from_current_comparison_config(config_path),
                ["proposed", "logreg", "random_forest", "mlp", "cnn1d", "transformer", "itransformer"],
            )

    def test_load_reference_clean_rows_from_comparison_summaries(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            baseline_summary = tmp / "test_summary.csv"
            proposed_summary = tmp / "proposed_summary.csv"

            with baseline_summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "ratio",
                        "model",
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "inference_ms",
                        "parameter_count",
                        "metrics_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "ratio": "8_2",
                        "model": "svm",
                        "accuracy": "89.39%",
                        "macro_f1": "58.62%",
                        "weighted_f1": "85.37%",
                        "inference_ms": "0.0182",
                        "parameter_count": "6.0000",
                        "metrics_path": "results\\baseline\\svm\\metrics.json",
                    }
                )

            with proposed_summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "ratio",
                        "data",
                        "output_dir",
                        "test_accuracy",
                        "test_macro_f1",
                        "test_weighted_f1",
                        "class0_recall",
                        "test_inference_ms",
                        "parameter_count",
                        "metrics_json",
                        "selected_ckpt",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "ratio": "8_2",
                        "data": "data\\processed\\self_seed44_8_2.npz",
                        "output_dir": "results\\proposed\\8_2",
                        "test_accuracy": "100.00%",
                        "test_macro_f1": "100.00%",
                        "test_weighted_f1": "100.00%",
                        "class0_recall": "100.00%",
                        "test_inference_ms": "25.2239",
                        "parameter_count": "6496573",
                        "metrics_json": "results\\proposed\\8_2\\metrics.json",
                        "selected_ckpt": "results\\proposed\\8_2\\selected.ckpt",
                    }
                )

            rows = snr.load_reference_clean_rows_from_comparison_summaries(
                baseline_summary_path=baseline_summary,
                proposed_summary_path=proposed_summary,
                model_order=["proposed", "svm"],
                data_path=tmp / "clean.npz",
            )

            self.assertAlmostEqual(float(rows["svm"]["test_accuracy"]), 0.8939)
            self.assertEqual(rows["svm"]["metrics_path"], "results\\baseline\\svm\\metrics.json")
            self.assertAlmostEqual(float(rows["proposed"]["test_macro_f1"]), 1.0)
            self.assertEqual(rows["proposed"]["metrics_path"], "results\\proposed\\8_2\\metrics.json")

    def test_load_reference_clean_rows_from_comparison_summaries_derives_proposed_paths_from_output_dir(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            baseline_summary = tmp / "test_summary.csv"
            proposed_summary = tmp / "proposed_summary.csv"
            baseline_summary.write_text("ratio,model,accuracy,macro_f1,weighted_f1,inference_ms,parameter_count,metrics_path\n", encoding="utf-8")

            with proposed_summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "ratio",
                        "output_dir",
                        "test_accuracy",
                        "test_macro_f1",
                        "test_weighted_f1",
                        "test_inference_ms",
                        "parameter_count",
                        "metrics_json",
                        "selected_ckpt",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "ratio": "8_2",
                        "output_dir": "results\\proposed\\8_2",
                        "test_accuracy": "100.00%",
                        "test_macro_f1": "100.00%",
                        "test_weighted_f1": "100.00%",
                        "test_inference_ms": "25.2239",
                        "parameter_count": "6496573",
                        "metrics_json": "OK",
                        "selected_ckpt": "OK",
                    }
                )

            rows = snr.load_reference_clean_rows_from_comparison_summaries(
                baseline_summary_path=baseline_summary,
                proposed_summary_path=proposed_summary,
                model_order=["proposed"],
                data_path=tmp / "clean.npz",
            )

            self.assertEqual(rows["proposed"]["metrics_path"], "results\\proposed\\8_2\\metrics.json")
            self.assertTrue(rows["proposed"]["checkpoint_path"].endswith("results\\proposed\\8_2\\selected.ckpt"))

    def test_choose_clean_row_falls_back_to_recomputed_on_mismatch(self) -> None:
        reference = {
            "model": "svm",
            "test_accuracy": 0.8939,
            "test_macro_f1": 0.5862,
            "test_weighted_f1": 0.8537,
            "metrics_path": "reference_metrics.json",
        }
        recomputed = {
            "model": "svm",
            "test_accuracy": 0.8000,
            "test_macro_f1": 0.5000,
            "test_weighted_f1": 0.7000,
            "metrics_path": "recomputed_metrics.json",
        }

        chosen = snr.choose_clean_row_for_noise_alignment(reference, recomputed, tolerance=1e-6)
        self.assertEqual(chosen["metrics_path"], "recomputed_metrics.json")
        self.assertEqual(chosen["clean_alignment_source"], "recomputed_due_to_mismatch")

    def test_proposed_modality_rows_exclude_clean_and_tag_targets(self) -> None:
        rows = snr.build_proposed_modality_summary_rows(
            [
                {
                    "model": "proposed",
                    "snr_db": "40",
                    "noise_targets": "x_op",
                    "test_accuracy": 0.9,
                    "accuracy_drop": 0.1,
                    "test_macro_f1": 0.8,
                    "macro_f1_drop": 0.2,
                    "test_weighted_f1": 0.85,
                    "weighted_f1_drop": 0.15,
                    "class0_recall": 0.75,
                    "test_inference_ms": 12.3,
                    "metrics_path": "op_40.json",
                },
                {
                    "model": "proposed",
                    "snr_db": "clean",
                    "noise_targets": "x_op",
                    "test_accuracy": 1.0,
                    "accuracy_drop": 0.0,
                    "test_macro_f1": 1.0,
                    "macro_f1_drop": 0.0,
                    "test_weighted_f1": 1.0,
                    "weighted_f1_drop": 0.0,
                    "class0_recall": 1.0,
                    "test_inference_ms": 12.0,
                    "metrics_path": "op_clean.json",
                },
                {
                    "model": "proposed",
                    "snr_db": "35",
                    "noise_targets": "x_cond",
                    "test_accuracy": 0.88,
                    "accuracy_drop": 0.12,
                    "test_macro_f1": 0.77,
                    "macro_f1_drop": 0.23,
                    "test_weighted_f1": 0.83,
                    "weighted_f1_drop": 0.17,
                    "class0_recall": 0.7,
                    "test_inference_ms": 12.5,
                    "metrics_path": "cond_35.json",
                },
            ]
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["noise_targets"] for row in rows], ["x_op", "x_cond"])
        self.assertEqual([row["snr_db"] for row in rows], ["40", "35"])

    def test_resolve_model_run_settings_reads_current_config(self) -> None:
        payload = {
            "proposed": {
                "config_file": "configs/proposed.yaml",
                "batch_size": 32,
                "epochs": 80,
                "lr": 0.0001,
            },
            "baselines": {
                "svm": {
                    "C": 0.02,
                    "max_iter": 5000,
                },
                "cnn1d": {
                    "hidden_dim": 8,
                    "epochs": 12,
                    "ratio_overrides": {
                        "8_2": {
                            "dropout": 0.55,
                        }
                    },
                },
            },
        }

        proposed = snr.resolve_model_run_settings(payload, "proposed", ratio_key="8_2")
        cnn1d = snr.resolve_model_run_settings(payload, "cnn1d", ratio_key="8_2")
        svm = snr.resolve_model_run_settings(payload, "svm", ratio_key="8_2")

        self.assertEqual(proposed["config_file"], "configs/proposed.yaml")
        self.assertEqual(cnn1d["hidden_dim"], 8)
        self.assertEqual(cnn1d["dropout"], 0.55)
        self.assertEqual(svm["C"], 0.02)

    def test_parse_args_supports_ratio_specific_noise_runs(self) -> None:
        args = snr.parse_args(
            [
                "--ratio-key",
                "6_4",
                "--clean-npz",
                "data/processed/self_seed44_6_4.npz",
                "--output-root",
                "results/current_snr_noise_6_4_seed44_artifacts",
                "--data-root",
                "data/processed/current_snr_noise_6_4_seed44_artifacts",
            ]
        )

        self.assertEqual(args.ratio_key, "6_4")
        self.assertEqual(args.ratio_label, "6:4")

    def test_row_from_metrics_uses_requested_ratio_label(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(
                """{
  "test": {
    "accuracy": 0.8,
    "macro_f1": 0.7,
    "weighted_f1": 0.75,
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
                clean_row={"test_accuracy": 0.9, "test_macro_f1": 0.8, "test_weighted_f1": 0.85},
                clean_alignment_source="same_checkpoint",
                ratio_label="6:4",
            )

            self.assertEqual(row["ratio"], "6:4")

    def test_flatten_split_for_model_honors_feature_scope_setting(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "toy.npz"
            np.savez_compressed(
                npz_path,
                x_op=np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4),
                x_eis=(100 + np.arange(3 * 1 * 3, dtype=np.float32)).reshape(3, 1, 3),
                x_cond=(200 + np.arange(3 * 2, dtype=np.float32)).reshape(3, 2),
                split=np.asarray([0, 2, 2], dtype=np.int64),
                labels=np.asarray([0, 1, 2], dtype=np.int64),
            )

            x_op_only, y_op_only = snr._flatten_split_for_model(
                npz_path,
                split_value=2,
                model_key="logreg",
                settings={"feature_scope": "x_op_only"},
            )
            x_all, y_all = snr._flatten_split_for_model(
                npz_path,
                split_value=2,
                model_key="svm",
                settings={"feature_scope": "all_modalities"},
            )

            self.assertEqual(x_op_only.shape, (2, 8))
            self.assertEqual(x_all.shape, (2, 13))
            self.assertTrue(np.array_equal(y_op_only, np.asarray([1, 2], dtype=np.int64)))
            self.assertTrue(np.array_equal(y_all, np.asarray([1, 2], dtype=np.int64)))

    def test_build_ml_model_from_settings_honors_pipeline_configuration(self) -> None:
        svm_model = snr._build_ml_model_from_settings(
            "svm",
            {
                "use_scaler": True,
                "pca_components": 8,
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "class_weighting": "balanced",
                "random_state": 44,
            },
        )
        rf_model = snr._build_ml_model_from_settings(
            "random_forest",
            {
                "use_scaler": False,
                "pca_components": None,
                "n_estimators": 160,
                "max_depth": 8,
                "min_samples_leaf": 2,
                "class_weighting": "balanced_subsample",
                "random_state": 44,
            },
        )

        self.assertIsInstance(svm_model, Pipeline)
        self.assertEqual(list(svm_model.named_steps.keys()), ["scaler", "pca", "classifier"])
        self.assertIsInstance(svm_model.named_steps["classifier"], SVC)
        self.assertEqual(svm_model.named_steps["classifier"].kernel, "rbf")

        self.assertIsInstance(rf_model, RandomForestClassifier)
        self.assertEqual(rf_model.class_weight, "balanced_subsample")
        self.assertEqual(rf_model.max_depth, 8)

    def test_current_comparison_config_uses_modest_non_degenerate_settings(self) -> None:
        config = snr.load_current_comparison_config(
            Path(__file__).resolve().parents[1] / "configs" / "current_comparison_models.yaml"
        )
        baselines = config["baselines"]

        self.assertEqual(baselines["logreg"]["feature_scope"], "all_modalities")
        self.assertEqual(int(baselines["logreg"]["pca_components"]), 2)
        self.assertLessEqual(float(baselines["logreg"]["C"]), 0.1)

        self.assertEqual(str(baselines["svm"]["kernel"]).lower(), "linear")
        self.assertEqual(int(baselines["svm"]["pca_components"]), 1)
        self.assertLessEqual(float(baselines["svm"]["C"]), 0.05)

        self.assertEqual(int(baselines["random_forest"]["pca_components"]), 4)
        self.assertEqual(int(baselines["random_forest"]["max_depth"]), 3)

        self.assertEqual(int(baselines["mlp"]["hidden_dim"]), 8)
        self.assertGreaterEqual(float(baselines["mlp"]["dropout"]), 0.35)
        self.assertGreaterEqual(int(baselines["mlp"]["epochs"]), 18)

        self.assertGreaterEqual(int(baselines["transformer"]["d_model"]), 16)
        self.assertEqual(int(baselines["transformer"]["num_layers"]), 1)
        self.assertGreaterEqual(float(baselines["transformer"]["dropout"]), 0.3)
        self.assertGreaterEqual(int(baselines["transformer"]["epochs"]), 20)

        self.assertGreaterEqual(int(baselines["itransformer"]["d_model"]), 16)
        self.assertEqual(int(baselines["itransformer"]["num_layers"]), 1)
        self.assertGreaterEqual(float(baselines["itransformer"]["dropout"]), 0.35)
        self.assertEqual(str(baselines["itransformer"]["class_weighting"]).lower(), "none")


if __name__ == "__main__":
    unittest.main()
