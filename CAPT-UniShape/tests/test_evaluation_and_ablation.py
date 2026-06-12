from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


class BaselineComparisonFigureTests(unittest.TestCase):
    def test_named_comparison_figure_script_writes_grouped_bar_chart(self) -> None:
        from experiments.utils.plot_results import plot_baseline_summary

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.csv"
            summary_path.write_text(
                "\n".join(
                    [
                        "ratio,model,test_macro_f1,test_accuracy",
                        "8_2,proposed,0.98,0.99",
                        "8_2,tcn,0.72,0.75",
                        "7_3,proposed,0.97,0.98",
                        "7_3,tcn,0.68,0.70",
                    ]
                ),
                encoding="utf-8",
            )

            output_path = plot_baseline_summary(summary_path, tmp_path)
            self.assertEqual(output_path.name, "official_baseline_comparison_summary.png")
            self.assertTrue(output_path.is_file())

    def test_baseline_summary_plot_merges_comparison_rows_from_workbook(self) -> None:
        from openpyxl import Workbook

        from experiments.utils.plot_results import _workbook_baseline_rows, plot_baseline_summary

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.csv"
            summary_path.write_text("ratio,model,test_macro_f1,test_accuracy\n8:2,proposed,1.0,1.0\n", encoding="utf-8")
            workbook_path = tmp_path / "results.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "对比实验"
            sheet.append(["数据集", "比例", "模型名称", "Test Acc", "Macro-F1", "Weighted-F1", "时间(ms/sample)", "参数量", "结果来源"])
            sheet.append(["自测数据集", "8:2", "SVM", "89.39%", "58.62%", "85.37%", 0.02, 6, "metrics.json"])
            workbook.save(workbook_path)

            rows = _workbook_baseline_rows(workbook_path)
            output_path = plot_baseline_summary(summary_path, tmp_path, results_workbook=workbook_path)

            self.assertEqual(rows[0]["model"], "svm")
            self.assertAlmostEqual(float(rows[0]["test_macro_f1"]), 0.5862)
            self.assertTrue(output_path.is_file())

    def test_baseline_summary_orders_ratios_from_small_to_large_training_share(self) -> None:
        from experiments.utils.plot_results import _read_summary_csv, _safe_float

        with TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.csv"
            summary_path.write_text(
                "\n".join(
                    [
                        "ratio,model,test_macro_f1,test_accuracy",
                        "8_2,proposed,1.0,1.0",
                        "5_5,proposed,0.90,0.90",
                        "5_5,svm,0.60,0.70",
                        "6_4,svm,0.61,0.71",
                    ]
                ),
                encoding="utf-8",
            )
            rows = _read_summary_csv(summary_path)
            ratio_order = {"5_5": 0, "6_4": 1, "7_3": 2, "8_2": 3}
            ratios = sorted({row["ratio"] for row in rows}, key=lambda item: ratio_order.get(item.replace(":", "_"), 99))
            reference_values = {
                row["model"]: _safe_float(row, "test_macro_f1")
                for row in rows
                if row.get("ratio", "").replace("_", ":") == "5:5"
            }
            models = sorted({row["model"] for row in rows}, key=lambda item: reference_values.get(item, float("inf")))

            self.assertEqual(ratios, ["5_5", "6_4", "8_2"])
            self.assertEqual(models, ["svm", "proposed"])

    def test_noise_summary_plot_accepts_snr_ablation_rows(self) -> None:
        from experiments.utils.plot_results import _read_summary_csv, plot_noise_summary

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "snr_summary.csv"
            summary_path.write_text(
                "\n".join(
                    [
                        "variant,scenario,snr_db,test_macro_f1",
                        "full_rbf,clean,clean,1.0",
                        "full_rbf,mid_noise,10,0.75",
                        "full_rbf,low_noise,30,0.91",
                        "full_rbf,high_noise,5,0.60",
                        "no_rbf,clean,clean,1.0",
                        "no_rbf,low_noise,30,0.88",
                        "no_rbf,high_noise,5,0.53",
                    ]
                ),
                encoding="utf-8",
            )

            output_path = plot_noise_summary(summary_path, tmp_path, results_workbook=None)
            self.assertEqual(output_path.name, "official_noise_summary.png")
            source_rows = _read_summary_csv(tmp_path / "official_noise_summary_source.csv")
            snr_order = list(dict.fromkeys(row["snr_db"] for row in source_rows))
            self.assertEqual(snr_order, ["5", "10", "30", "clean"])
            self.assertTrue(output_path.is_file())

    def test_noise_summary_plot_uses_workbook_comparison_models_only(self) -> None:
        from openpyxl import Workbook

        from experiments.utils.plot_results import _read_summary_csv, _workbook_noise_rows, plot_noise_summary

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "snr_summary.csv"
            summary_path.write_text(
                "\n".join(
                    [
                        "variant,scenario,snr_db,test_macro_f1,test_accuracy,test_weighted_f1,macro_f1,accuracy,weighted_f1",
                        "full_rbf,clean,clean,1.0,1.0,1.0,,,",
                        "full_rbf,low_noise,30,0.91,0.92,0.93,,,",
                        "xgboost,clean,clean,,,,0.82,0.83,0.84",
                        "xgboost,low_noise,30,,,,0.72,0.73,0.74",
                    ]
                ),
                encoding="utf-8",
            )
            workbook_path = tmp_path / "results.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "SNR噪声对比"
            sheet.append(["model", "snr_db", "test_accuracy", "accuracy_drop", "test_macro_f1", "macro_f1_drop"])
            sheet.append(["proposed", 30, 0.50, 0, 0.50, 0])
            sheet.append(["svm", "clean", 0.8934, 0, 0.875, 0])
            sheet.append(["svm", 30, 0.8962, -0.0027, 0.8776, -0.0026])
            workbook.save(workbook_path)

            rows = _workbook_noise_rows(workbook_path)
            output_path = plot_noise_summary(summary_path, tmp_path, results_workbook=workbook_path)

            self.assertEqual({row["variant"] for row in rows}, {"proposed", "svm"})
            merged_rows = _read_summary_csv(tmp_path / "official_noise_summary_source.csv")
            self.assertEqual({row["variant"] for row in merged_rows}, {"proposed", "svm", "xgboost"})
            self.assertAlmostEqual(
                float([row for row in merged_rows if row["variant"] == "proposed" and row["snr_db"] == "30"][-1]["test_macro_f1"]),
                0.91,
            )
            self.assertAlmostEqual(
                float([row for row in merged_rows if row["variant"] == "xgboost" and row["snr_db"] == "30"][-1]["macro_f1"]),
                0.72,
            )
            self.assertTrue(output_path.is_file())

    def test_metric_reader_falls_back_to_compact_summary_columns(self) -> None:
        from experiments.utils.plot_results import _safe_metric_float

        row = {
            "test_macro_f1": "",
            "macro_f1": "0.7448",
            "test_accuracy": "",
            "accuracy": "0.7917",
            "test_weighted_f1": "",
            "weighted_f1": "0.7951",
        }

        self.assertAlmostEqual(_safe_metric_float(row, "test_macro_f1"), 0.7448)
        self.assertAlmostEqual(_safe_metric_float(row, "test_accuracy"), 0.7917)
        self.assertAlmostEqual(_safe_metric_float(row, "test_weighted_f1"), 0.7951)

    def test_noise_summary_orders_full_snr_axis_and_reads_compact_baseline_metrics(self) -> None:
        from experiments.utils.plot_results import _read_summary_csv, _safe_metric_float, plot_noise_summary

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "snr_summary.csv"
            rows = [
                "variant,snr_db,test_accuracy,test_macro_f1,test_weighted_f1,accuracy,macro_f1,weighted_f1",
            ]
            for snr in ["40", "35", "30", "25", "20", "15", "10", "5", "clean"]:
                rows.append(f"full_rbf,{snr},0.98,0.97,0.98,,,")
                rows.append(f"xgboost,{snr},,,,0.79,0.74,0.78")
            summary_path.write_text("\n".join(rows), encoding="utf-8")

            output_path = plot_noise_summary(summary_path, tmp_path, results_workbook=None)

            source_rows = _read_summary_csv(tmp_path / "official_noise_summary_source.csv")
            snr_order = list(dict.fromkeys(row["snr_db"] for row in source_rows))
            self.assertEqual(snr_order, ["5", "10", "15", "20", "25", "30", "35", "40", "clean"])
            baseline_5db = next(row for row in source_rows if row["variant"] == "xgboost" and row["snr_db"] == "5")
            self.assertEqual(baseline_5db["test_accuracy"], "")
            self.assertEqual(baseline_5db["test_macro_f1"], "")
            self.assertEqual(baseline_5db["test_weighted_f1"], "")
            self.assertAlmostEqual(_safe_metric_float(baseline_5db, "test_accuracy"), 0.79)
            self.assertAlmostEqual(_safe_metric_float(baseline_5db, "test_macro_f1"), 0.74)
            self.assertAlmostEqual(_safe_metric_float(baseline_5db, "test_weighted_f1"), 0.78)
            self.assertTrue(output_path.is_file())


class EvaluationSplitTests(unittest.TestCase):
    def test_split_indices_from_npz_selects_only_requested_split(self) -> None:
        from evaluate import split_indices_from_npz

        data = {"split": np.array([0, 1, 2, 2, 1], dtype=np.int64)}

        self.assertEqual(split_indices_from_npz(data, "test").tolist(), [2, 3])
        self.assertEqual(split_indices_from_npz(data, "val").tolist(), [1, 4])
        self.assertEqual(split_indices_from_npz(data, "all").tolist(), [0, 1, 2, 3, 4])

    def test_split_indices_from_npz_rejects_missing_split_for_named_split(self) -> None:
        from evaluate import split_indices_from_npz

        with self.assertRaises(ValueError):
            split_indices_from_npz({}, "test")

    def test_evaluation_artifact_semantics_use_requested_split_name(self) -> None:
        from evaluate import evaluation_artifact_semantics

        semantics = evaluation_artifact_semantics("test")

        self.assertEqual(semantics["evaluated_split"], "test")
        self.assertEqual(semantics["top_level_metrics"], "test")


class ResidualKANFusionTests(unittest.TestCase):
    def test_disabled_kan_fusion_returns_zero_kan_regularization(self) -> None:
        from models.modules.residual_kan_fusion import ResidualKANFusion

        torch.manual_seed(7)
        module = ResidualKANFusion(
            input_dim=6,
            d_model=4,
            hidden_dim=8,
            bottleneck_dim=3,
            num_basis=4,
            use_residual_kan=False,
        )
        output, aux = module(torch.randn(5, 6))

        self.assertEqual(tuple(output.shape), (5, 4))
        self.assertAlmostEqual(float(aux["kan_regularization"].detach().cpu()), 0.0, places=7)

    def test_disabled_kan_fusion_excludes_unused_kan_parameters(self) -> None:
        from models.modules.residual_kan_fusion import ResidualKANFusion

        enabled = ResidualKANFusion(input_dim=6, d_model=4, hidden_dim=8, bottleneck_dim=3, num_basis=4, use_residual_kan=True)
        disabled = ResidualKANFusion(input_dim=6, d_model=4, hidden_dim=8, bottleneck_dim=3, num_basis=4, use_residual_kan=False)

        enabled_params = sum(parameter.numel() for parameter in enabled.parameters())
        disabled_params = sum(parameter.numel() for parameter in disabled.parameters())

        self.assertLess(disabled_params, enabled_params)


class AblationSwitchTests(unittest.TestCase):
    def test_official_ablation_default_variants_match_required_table(self) -> None:
        from experiments.run_official_ablation_experiments import ABLATIONS, DEFAULT_ABLATION_VARIANTS

        self.assertEqual(
            DEFAULT_ABLATION_VARIANTS,
            ["full_rbf", "no_rbf", "no_kan_fusion", "static_prototype", "no_condition_input"],
        )
        self.assertEqual({spec["config"] for spec in ABLATIONS.values()}, {"configs/ablation.yaml"})
        self.assertEqual(ABLATIONS["no_rbf"]["overrides"]["model_name"], "official_capt_unishape_kanfusion_no_rbf")
        self.assertFalse(ABLATIONS["no_rbf"]["overrides"]["use_rbf_head"])

    def test_official_ablation_variants_use_capacity_reduced_overrides(self) -> None:
        from experiments.run_official_ablation_experiments import ABLATIONS

        self.assertNotIn("overrides", ABLATIONS["full_rbf"])

        expected = {
            "no_rbf": {
                "use_rbf_head": False,
                "hidden_dim": 32,
                "fusion_hidden_dim": 32,
                "dropout": 0.45,
            },
            "no_kan_fusion": {
                "use_residual_kan_fusion": False,
                "hidden_dim": 64,
                "fusion_hidden_dim": 32,
                "dropout": 0.45,
            },
            "static_prototype": {
                "use_condition_transport": False,
                "hidden_dim": 64,
                "fusion_hidden_dim": 32,
                "num_rbf_centers": 2,
                "dropout": 0.45,
            },
            "no_condition_input": {
                "hidden_dim": 32,
                "fusion_hidden_dim": 32,
                "dropout": 0.5,
            },
        }
        for variant, expected_overrides in expected.items():
            with self.subTest(variant=variant):
                overrides = ABLATIONS[variant].get("overrides", {})
                for key, value in expected_overrides.items():
                    self.assertEqual(overrides[key], value)

    def test_official_ablation_metric_row_excludes_class0_fields(self) -> None:
        from experiments.run_official_ablation_experiments import _copy_existing_metric_row

        with TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(
                """
{
  "accuracy": 1.0,
  "macro_f1": 0.9,
  "weighted_f1": 0.95,
  "inference_time_per_sample_ms": 12.34567,
  "classification_report": {
    "0": {"precision": 1.0, "recall": 0.75, "f1-score": 0.857142}
  },
  "parameter_count": 1234,
  "artifact_semantics": {"top_level_metrics": "test"}
}
""",
                encoding="utf-8",
            )

            row = _copy_existing_metric_row("full_rbf", "完整模型", metrics_path)

        self.assertEqual(row["variant"], "full_rbf")
        self.assertAlmostEqual(row["test_accuracy"], 1.0)
        self.assertAlmostEqual(row["test_macro_f1"], 0.9)
        self.assertEqual(row["parameter_count"], 1234)
        self.assertNotIn("class0_precision", row)
        self.assertNotIn("class0_recall", row)
        self.assertNotIn("class0_f1", row)

    def test_official_ablation_default_data_points_to_updated_self_dataset(self) -> None:
        from experiments.run_official_ablation_experiments import parse_args

        args = parse_args([])

        self.assertEqual(args.data, "data/processed/self_seed44_8_2.npz")

    def test_snr_eval_args_default_to_current_six_four_artifacts(self) -> None:
        from experiments.run_official_ablation_experiments import parse_args

        args = parse_args(["--snr-eval-only"])

        self.assertTrue(args.snr_eval_only)
        self.assertEqual(args.reuse_checkpoints_root, "results/current_ablation_updated_dataset_seed44_6_4")
        self.assertEqual(args.snr_output_root, "results/current_ablation_snr_updated_dataset_seed44_6_4")
        self.assertEqual(args.snr_dbs, [30.0, 5.0])
        self.assertEqual(args.snr_noise_seeds, [44, 45, 46])

    def test_ablation_snr_scenario_labels_are_explicit(self) -> None:
        from experiments.run_official_ablation_experiments import _scenario_label

        self.assertEqual(_scenario_label("clean"), "clean")
        self.assertEqual(_scenario_label("30"), "low_noise")
        self.assertEqual(_scenario_label("5"), "high_noise")

    def test_mean_snr_rows_average_only_noisy_repeats(self) -> None:
        from experiments.run_official_ablation_experiments import _mean_snr_rows

        rows = [
            {
                "variant": "full_rbf",
                "description": "完整模型",
                "scenario": "clean",
                "snr_db": "clean",
                "noise_seed": "",
                "actual_snr_db_mean": "",
                "noise_targets": "x_op+x_eis+x_cond",
                "zeroed_inputs": "",
                "test_accuracy": 1.0,
                "accuracy_drop": 0.0,
                "test_macro_f1": 1.0,
                "macro_f1_drop": 0.0,
                "test_weighted_f1": 1.0,
                "weighted_f1_drop": 0.0,
                "test_inference_ms": 10.0,
                "parameter_count": 123,
                "data_path": "clean.npz",
                "metrics_path": "clean/metrics.json",
            },
            {
                "variant": "full_rbf",
                "description": "完整模型",
                "scenario": "low_noise",
                "snr_db": "30",
                "noise_seed": 44,
                "actual_snr_db_mean": 30.0,
                "noise_targets": "x_op+x_eis+x_cond",
                "zeroed_inputs": "",
                "test_accuracy": 0.9,
                "accuracy_drop": 0.1,
                "test_macro_f1": 0.8,
                "macro_f1_drop": 0.2,
                "test_weighted_f1": 0.85,
                "weighted_f1_drop": 0.15,
                "test_inference_ms": 10.0,
                "parameter_count": 123,
                "data_path": "seed44.npz",
                "metrics_path": "seed44/metrics.json",
            },
            {
                "variant": "full_rbf",
                "description": "完整模型",
                "scenario": "low_noise",
                "snr_db": "30",
                "noise_seed": 45,
                "actual_snr_db_mean": 30.0,
                "noise_targets": "x_op+x_eis+x_cond",
                "zeroed_inputs": "",
                "test_accuracy": 0.7,
                "accuracy_drop": 0.3,
                "test_macro_f1": 0.6,
                "macro_f1_drop": 0.4,
                "test_weighted_f1": 0.65,
                "weighted_f1_drop": 0.35,
                "test_inference_ms": 14.0,
                "parameter_count": 123,
                "data_path": "seed45.npz",
                "metrics_path": "seed45/metrics.json",
            },
        ]

        mean_rows = _mean_snr_rows(rows)

        self.assertEqual(len(mean_rows), 2)
        self.assertEqual(mean_rows[0]["snr_db"], "clean")
        self.assertEqual(mean_rows[1]["scenario"], "low_noise")
        self.assertEqual(mean_rows[1]["noise_seed"], "mean")
        self.assertEqual(mean_rows[1]["n_noise_seeds"], 2)
        self.assertAlmostEqual(mean_rows[1]["test_accuracy"], 0.8)
        self.assertAlmostEqual(mean_rows[1]["test_macro_f1"], 0.7)
        self.assertAlmostEqual(mean_rows[1]["test_inference_ms"], 12.0)

    def test_snr_noise_targets_skip_zeroed_ablation_inputs(self) -> None:
        from experiments.run_official_ablation_experiments import ABLATIONS, _active_noise_targets

        requested = ["x_op", "x_eis", "x_cond"]

        self.assertEqual(_active_noise_targets(ABLATIONS["full_rbf"], requested), requested)
        self.assertEqual(_active_noise_targets(ABLATIONS["no_condition_input"], requested), ["x_op", "x_eis"])

    def test_fixed_equal_fusion_replaces_condition_gates(self) -> None:
        from models.capt_unishape_kanfusion_no_rbf import OfficialCAPTUniShapeKANFusionNoRBF

        model = OfficialCAPTUniShapeKANFusionNoRBF(use_condition_gating=False)
        z_op = torch.tensor([[1.0, 2.0]])
        z_eis = torch.tensor([[3.0, 5.0]])
        z_cond = torch.tensor([[7.0, 11.0]])

        z_fused, g_op, g_eis = model._fuse_modal_features(z_op, z_eis, z_cond)

        self.assertTrue(torch.equal(z_fused, z_op + z_eis + z_cond))
        self.assertTrue(torch.equal(g_op, torch.ones_like(z_op)))
        self.assertTrue(torch.equal(g_eis, torch.ones_like(z_eis)))


class OfficialNoiseExperimentCLITests(unittest.TestCase):
    def test_noise_experiment_defaults_to_five_five_official_run(self) -> None:
        from experiments.run_official_noise_experiments import MODELS, parse_args

        args = parse_args([])

        self.assertEqual(args.ratio, "5_5")
        self.assertEqual(args.models, ["proposed"])
        self.assertEqual(args.seed, 44)
        self.assertIn("noise_5_5", args.output_root)
        self.assertIn("noise_5_5", args.data_root)
        self.assertEqual(set(MODELS), {"proposed"})
        self.assertEqual(MODELS["proposed"], "configs/proposed.yaml")

    def test_noise_experiment_defaults_to_snr_five_to_forty_by_five(self) -> None:
        from experiments.run_official_noise_experiments import parse_args

        args = parse_args([])

        self.assertEqual(args.snr_dbs, [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
        self.assertEqual(args.snr_scope, "per_sample_modality")
        self.assertIn("snr", args.output_root)
        self.assertIn("snr", args.data_root)

    def test_snr_metric_row_uses_plot_results_schema(self) -> None:
        from experiments.run_official_noise_experiments import _snr_metric_row

        with TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(
                '{"accuracy": 0.9, "macro_f1": 0.8, "weighted_f1": 0.85, "inference_time_per_sample_ms": 12.3, "parameter_count": 7}',
                encoding="utf-8",
            )

            row = _snr_metric_row("5_5", "proposed", "40", 40.0, ["x_op"], Path("snr_40dB.npz"), metrics_path)

        self.assertEqual(row["ratio"], "5:5")
        self.assertEqual(row["variant"], "proposed")
        self.assertEqual(row["model"], "proposed")
        self.assertEqual(row["scenario"], "snr_40dB")
        self.assertEqual(row["snr_db"], "40")
        self.assertNotIn("noise_std", row)


class TrainingUtilityTests(unittest.TestCase):
    def test_snapshot_state_dict_clones_parameter_values(self) -> None:
        from train import snapshot_state_dict

        layer = torch.nn.Linear(2, 1, bias=False)
        snapshot = snapshot_state_dict(layer)
        original_value = snapshot["weight"].clone()

        with torch.no_grad():
            layer.weight.add_(10.0)

        self.assertTrue(torch.equal(snapshot["weight"], original_value))
        self.assertFalse(torch.equal(snapshot["weight"], layer.state_dict()["weight"].cpu()))

    def test_class_logit_adjustment_uses_training_class_counts(self) -> None:
        from train import compute_class_logit_adjustment

        counts = torch.tensor([2.0, 6.0, 12.0])
        adjustment = compute_class_logit_adjustment(counts, tau=1.0)

        expected = torch.log(counts / counts.sum())
        self.assertTrue(torch.allclose(adjustment.cpu(), expected))

    def test_apply_experiment_overrides_updates_lr_and_weight_decay(self) -> None:
        from train import apply_experiment_overrides

        experiment = {"lr": 1e-4, "weight_decay": 1e-4, "batch_size": 16}
        updated = apply_experiment_overrides(experiment, lr=3e-4, weight_decay=5e-5)

        self.assertEqual(updated["batch_size"], 16)
        self.assertAlmostEqual(updated["lr"], 3e-4)
        self.assertAlmostEqual(updated["weight_decay"], 5e-5)
        self.assertAlmostEqual(experiment["lr"], 1e-4)

    def test_attach_effective_experiment_settings_updates_saved_config(self) -> None:
        from train import attach_effective_experiment_settings

        config = {"experiment": {"lr": 1e-4}}
        experiment = {"lr": 3e-4, "weight_decay": 5e-5}

        updated = attach_effective_experiment_settings(config, experiment)

        self.assertIs(updated, config)
        self.assertAlmostEqual(updated["experiment"]["lr"], 3e-4)
        self.assertAlmostEqual(updated["experiment"]["weight_decay"], 5e-5)

    def test_normalize_checkpoint_selection_accepts_last_strategy(self) -> None:
        from train import normalize_checkpoint_selection

        self.assertEqual(normalize_checkpoint_selection("last"), "last")
        self.assertEqual(normalize_checkpoint_selection("best-val"), "best_val")

    def test_checkpoint_alias_names_include_selected_and_strategy_specific_name(self) -> None:
        from train import checkpoint_alias_names

        self.assertEqual(checkpoint_alias_names("best_val"), ["best.ckpt", "selected.ckpt"])
        self.assertEqual(checkpoint_alias_names("last"), ["best.ckpt", "selected.ckpt", "last.ckpt"])


class ProposedAccuracySearchTests(unittest.TestCase):
    def test_metric_record_prefers_authoritative_test_metrics(self) -> None:
        from scripts.run_proposed_accuracy_search import metric_record_from_payload

        record = metric_record_from_payload(
            Path("results/example/metrics.json"),
            {
                "accuracy": 0.50,
                "macro_f1": 0.40,
                "test": {"accuracy": 0.97, "macro_f1": 0.96},
                "split_diagnostics": {"test_size": 12},
            },
            source="candidate",
        )

        self.assertAlmostEqual(record["test_accuracy"], 0.97)
        self.assertAlmostEqual(record["test_macro_f1"], 0.96)
        self.assertEqual(record["metric_source"], "test")
        self.assertEqual(record["split_diagnostics"]["test_size"], 12)

    def test_first_successful_record_uses_validation_metric_by_default(self) -> None:
        from scripts.run_proposed_accuracy_search import first_successful_record

        records = [
            {"test_accuracy": 0.99, "val_macro_f1": 0.80, "metrics_path": "a"},
            {"test_accuracy": 0.90, "val_macro_f1": 0.95, "metrics_path": "b"},
            {"test_accuracy": 0.99, "val_macro_f1": 0.99, "metrics_path": "c"},
        ]

        selected = first_successful_record(records, threshold=0.95)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["metrics_path"], "b")

    def test_search_args_default_stop_metric_is_validation_not_test(self) -> None:
        from scripts.run_proposed_accuracy_search import parse_args

        args = parse_args(["--dry-run"])

        self.assertEqual(args.search_metric, "val_macro_f1")

    def test_search_command_forwards_checkpoint_selection(self) -> None:
        from scripts.run_proposed_accuracy_search import attempt_command, build_attempts, parse_args

        args = parse_args(["--checkpoint-selection", "last", "--max-attempts", "1", "--dry-run"])
        attempt = build_attempts(args)[0]
        command = attempt_command(args, attempt)

        self.assertIn("--checkpoint-selection", command)
        self.assertIn("last", command)


class MultiSeedExperimentCLITests(unittest.TestCase):
    def test_multiseed_cli_accepts_logit_adjusted_class_weighting(self) -> None:
        from scripts.run_official_multiseed_experiments import parse_args

        args = parse_args(["--class-weighting", "logit_adjusted", "--skip-run"])

        self.assertEqual(args.class_weighting, "logit_adjusted")


if __name__ == "__main__":
    unittest.main()

