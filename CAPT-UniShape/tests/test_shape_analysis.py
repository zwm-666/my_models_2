from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class ShapeAnalysisTests(unittest.TestCase):
    def test_panel_label_style_uses_requested_position_without_bold(self) -> None:
        from experiments.plot_shape_analysis import PANEL_LABEL_KWARGS

        self.assertEqual(PANEL_LABEL_KWARGS["x"], 0.02)
        self.assertEqual(PANEL_LABEL_KWARGS["y"], 1.08)
        self.assertEqual(PANEL_LABEL_KWARGS["fontweight"], "normal")

    def test_shape_descriptors_capture_basic_curve_properties(self) -> None:
        from experiments.plot_shape_analysis import shape_descriptors

        x_eis = np.array(
            [
                [
                    [1.0, 2.0, 4.0, 8.0],
                    [0.0, 1.0, 2.0, 4.0],
                    [-1.0, -0.2, 0.2, 1.0],
                    [0.0, 0.33, 0.66, 1.0],
                ]
            ],
            dtype=np.float32,
        )

        rows = shape_descriptors(x_eis, labels=np.array([2], dtype=np.int64), split_name="test")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], 2)
        self.assertEqual(rows[0]["class_name"], "过干")
        self.assertEqual(rows[0]["split"], "test")
        self.assertAlmostEqual(rows[0]["curve_mean"], 3.75, places=4)
        self.assertAlmostEqual(rows[0]["curve_range"], 7.0, places=4)
        self.assertAlmostEqual(rows[0]["gradient_energy"], 21.0, places=4)
        self.assertAlmostEqual(rows[0]["cumulative_range"], 2.0, places=4)
        self.assertAlmostEqual(rows[0]["peak_position"], 1.0, places=4)
        self.assertAlmostEqual(rows[0]["trough_position"], 0.0, places=4)

    def test_classwise_mean_std_keeps_present_classes(self) -> None:
        from experiments.plot_shape_analysis import classwise_mean_std

        curves = np.array(
            [
                [1.0, 3.0],
                [3.0, 5.0],
                [10.0, 14.0],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 2], dtype=np.int64)

        stats = classwise_mean_std(curves, labels)

        self.assertEqual(sorted(stats), [0, 2])
        self.assertTrue(np.allclose(stats[0]["mean"], [2.0, 4.0]))
        self.assertTrue(np.allclose(stats[0]["std"], [1.0, 1.0]))
        self.assertTrue(np.allclose(stats[2]["mean"], [10.0, 14.0]))
        self.assertTrue(np.allclose(stats[2]["std"], [0.0, 0.0]))
        self.assertEqual(stats[2]["count"], 1)

    def test_write_descriptor_csv_formats_numbers_consistently(self) -> None:
        from experiments.plot_shape_analysis import write_descriptor_csv

        rows = [
            {
                "split": "test",
                "label": 1,
                "class_name": "过湿",
                "curve_mean": 1.23456,
                "curve_std": 0.1,
                "curve_range": 2.0,
                "gradient_energy": 3.33333,
                "gradient_abs_mean": 0.25,
                "cumulative_range": 1.0,
                "peak_position": 0.75,
                "trough_position": 0.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "descriptors.csv"
            write_descriptor_csv(rows, out)

            text = out.read_text(encoding="utf-8-sig")

        self.assertIn("curve_mean", text)
        self.assertIn("1.2346", text)
        self.assertIn("3.3333", text)

    def test_channel_replacement_preserves_other_shape_channels(self) -> None:
        from experiments.plot_shape_importance import replace_channel_with_reference

        x_eis = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        reference = np.full((4, 3), -1.0, dtype=np.float32)

        replaced = replace_channel_with_reference(x_eis, channel=2, reference=reference)

        self.assertTrue(np.array_equal(replaced[:, 0, :], x_eis[:, 0, :]))
        self.assertTrue(np.array_equal(replaced[:, 1, :], x_eis[:, 1, :]))
        self.assertTrue(np.array_equal(replaced[:, 3, :], x_eis[:, 3, :]))
        self.assertTrue(np.array_equal(replaced[:, 2, :], np.full((2, 3), -1.0, dtype=np.float32)))

    def test_metric_drop_rows_are_nonnegative_when_score_decreases(self) -> None:
        from experiments.plot_shape_importance import metric_drop_row

        row = metric_drop_row(
            channel=1,
            channel_name="gradient",
            baseline={"accuracy": 0.9, "macro_f1": 0.8, "mean_true_probability": 0.7},
            perturbed={"accuracy": 0.75, "macro_f1": 0.65, "mean_true_probability": 0.6},
            method="mean_replacement",
        )

        self.assertEqual(row["channel"], 1)
        self.assertEqual(row["channel_name"], "gradient")
        self.assertEqual(row["method"], "mean_replacement")
        self.assertAlmostEqual(row["accuracy_drop"], 0.15, places=4)
        self.assertAlmostEqual(row["macro_f1_drop"], 0.15, places=4)
        self.assertAlmostEqual(row["mean_true_probability_drop"], 0.10, places=4)

    def test_classwise_mean_abs_impact_returns_feature_by_class_matrix(self) -> None:
        from experiments.plot_eis_shap_style import classwise_mean_abs_impact

        impacts = np.array(
            [
                [[1.0, -2.0], [0.5, 1.0], [0.0, 3.0]],
                [[3.0, 4.0], [1.5, -1.0], [2.0, -1.0]],
            ],
            dtype=np.float32,
        )

        matrix = classwise_mean_abs_impact(impacts)

        self.assertEqual(matrix.shape, (3, 2))
        self.assertTrue(np.allclose(matrix[0], [2.0, 3.0]))
        self.assertTrue(np.allclose(matrix[1], [1.0, 1.0]))
        self.assertTrue(np.allclose(matrix[2], [1.0, 2.0]))

    def test_top_feature_indices_sort_by_total_importance(self) -> None:
        from experiments.plot_eis_shap_style import top_feature_indices

        matrix = np.array(
            [
                [0.1, 0.2, 0.0],
                [0.5, 0.1, 0.1],
                [0.2, 0.2, 0.1],
            ],
            dtype=np.float32,
        )

        self.assertEqual(top_feature_indices(matrix, top_k=2).tolist(), [1, 2])

    def test_normalize_shap_values_accepts_multiclass_list_output(self) -> None:
        from experiments.plot_eis_shap_style import normalize_shap_values

        values = [
            np.ones((2, 3), dtype=np.float32) * 1.0,
            np.ones((2, 3), dtype=np.float32) * 2.0,
            np.ones((2, 3), dtype=np.float32) * 3.0,
        ]

        out = normalize_shap_values(values)

        self.assertEqual(out.shape, (2, 3, 3))
        self.assertTrue(np.allclose(out[:, :, 0], 1.0))
        self.assertTrue(np.allclose(out[:, :, 2], 3.0))

    def test_eis_display_names_are_english(self) -> None:
        from experiments.plot_eis_shap_style import english_feature_names

        names = english_feature_names(["总阻抗", "EIS电阻实部", "EIS电阻虚部"])

        self.assertEqual(names, ["Total impedance", "EIS resistance real", "EIS resistance imaginary"])
        self.assertTrue(all(all(ord(ch) < 128 for ch in name) for name in names))

    def test_operational_display_names_are_english(self) -> None:
        from experiments.plot_eis_shap_style import english_feature_names

        names = english_feature_names(["电堆总电压", "电堆总电流", "电堆功率"])

        self.assertEqual(names, ["Stack voltage", "Stack current", "Stack power"])
        self.assertTrue(all(all(ord(ch) < 128 for ch in name) for name in names))

    def test_shap_condition_display_names_are_english(self) -> None:
        from experiments.plot_eis_shap_style import english_feature_names

        names = english_feature_names(["进堆空压", "氢压差", "FC空压机出口温度"])

        self.assertEqual(names, ["Air inlet pressure", "Hydrogen pressure drop", "Compressor outlet temperature"])
        self.assertTrue(all(all(ord(ch) < 128 for ch in name) for name in names))

    def test_feature_order_by_total_mean_abs_shap(self) -> None:
        from experiments.plot_eis_shap_style import feature_order_by_total_impact

        matrix = np.array(
            [
                [0.1, 0.0, 0.0],
                [0.1, 0.2, 0.4],
                [0.3, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.assertEqual(feature_order_by_total_impact(matrix, top_k=2).tolist(), [1, 2])

    def test_condition_candidate_columns_exclude_model_input_sources(self) -> None:
        import pandas as pd

        from experiments.plot_condition_selection import condition_candidate_columns

        frame = pd.DataFrame(
            {
                "测试时间": ["2026-01-01"],
                "label": [0],
                "总阻抗": [1.0],
                "电堆总电压": [2.0],
                "进堆空压": [3.0],
                "出堆水温": [4.0],
                "__label__": [0],
                "note": ["x"],
            }
        )

        self.assertEqual(condition_candidate_columns(frame), ["进堆空压", "出堆水温"])

    def test_condition_display_names_are_english(self) -> None:
        from experiments.plot_condition_selection import english_condition_names

        names = english_condition_names(["进堆空压", "氢压差", "FC空压机出口温度"])

        self.assertEqual(names, ["Air inlet pressure", "Hydrogen pressure drop", "Compressor outlet temperature"])
        self.assertTrue(all(all(ord(ch) < 128 for ch in name) for name in names))


if __name__ == "__main__":
    unittest.main()
