from __future__ import annotations

import importlib.util
import unittest

import numpy as np


class FeatureEmbeddingVisualizationTests(unittest.TestCase):
    def test_split_indices_from_npz_selects_named_split(self) -> None:
        from scripts.plot_feature_embeddings import split_indices_from_npz

        data = {"split": np.array([0, 2, 1, 2, 0], dtype=np.int64)}

        self.assertEqual(split_indices_from_npz(data, "train").tolist(), [0, 4])
        self.assertEqual(split_indices_from_npz(data, "val").tolist(), [2])
        self.assertEqual(split_indices_from_npz(data, "test").tolist(), [1, 3])
        self.assertEqual(split_indices_from_npz(data, "all").tolist(), [0, 1, 2, 3, 4])

    def test_raw_feature_matrix_concatenates_modalities_for_same_indices(self) -> None:
        from scripts.plot_feature_embeddings import raw_feature_matrix

        data = {
            "x_op": np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3),
            "x_eis": np.arange(2 * 1 * 2, dtype=np.float32).reshape(2, 1, 2) + 100,
            "x_cond": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        }

        features = raw_feature_matrix(data, np.array([1], dtype=np.int64))

        self.assertEqual(features.shape, (1, 10))
        expected = np.concatenate(
            [
                data["x_op"][1].reshape(-1),
                data["x_eis"][1].reshape(-1),
                data["x_cond"][1].reshape(-1),
            ]
        )
        self.assertTrue(np.array_equal(features[0], expected))

    def test_reduce_features_keeps_two_columns_with_small_sample_count(self) -> None:
        from scripts.plot_feature_embeddings import reduce_features

        features = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.2, 0.0],
                [0.1, 1.0, 0.3],
            ],
            dtype=np.float32,
        )

        reduced = reduce_features(features, method="pca", seed=7)

        self.assertEqual(reduced.shape, (3, 2))
        self.assertTrue(np.isfinite(reduced).all())

    def test_axis_labels_name_the_embedding_method_not_fake_raw_features(self) -> None:
        from scripts.plot_feature_embeddings import embedding_axis_labels

        self.assertEqual(embedding_axis_labels("tsne"), ("t-SNE 1", "t-SNE 2"))
        self.assertEqual(embedding_axis_labels("umap"), ("UMAP 1", "UMAP 2"))
        self.assertEqual(embedding_axis_labels("pca"), ("PC1", "PC2"))

    def test_panel_titles_describe_real_feature_sources(self) -> None:
        from scripts.plot_feature_embeddings import panel_title

        self.assertEqual(panel_title("raw"), "(a) Raw input features")
        self.assertEqual(panel_title("raw_source"), "(a) Raw source features")
        self.assertEqual(panel_title("baseline", "transformer"), "(b) Baseline model features")
        self.assertEqual(panel_title("unishape_before"), "(c) UniShape features before shape-aware adapter")
        self.assertEqual(panel_title("unishape_after"), "(d) UniShape final features after shape-aware adapter")

    def test_raw_self_window_features_keep_excel_physical_values(self) -> None:
        import pandas as pd

        from scripts.plot_feature_embeddings import build_raw_self_window_feature_arrays

        stack_cols = ["电堆总电压", "电堆总电流", "电堆功率"]
        cond_cols = [
            "总阻抗",
            "平均阻抗",
            "最高阻抗",
            "次高阻抗",
            "最低阻抗",
            "次低阻抗",
            "标准差",
            "EIS电阻实部",
            "EIS电阻虚部",
            *stack_cols,
        ]
        frame = pd.DataFrame(
            {
                "测试时间": pd.date_range("2026-01-01", periods=3, freq="min"),
                "类型": [1, 1, 1],
                "电堆总电压": [100.0, 102.0, 104.0],
                "电堆总电流": [10.0, 12.0, 14.0],
                "电堆功率": [1.0, 2.0, 3.0],
                "总阻抗": [50.0, 60.0, 70.0],
                "平均阻抗": [5.0, 6.0, 7.0],
                "最高阻抗": [8.0, 9.0, 10.0],
                "次高阻抗": [7.0, 8.0, 9.0],
                "最低阻抗": [1.0, 2.0, 3.0],
                "次低阻抗": [2.0, 3.0, 4.0],
                "标准差": [0.1, 0.2, 0.3],
                "EIS电阻实部": [0.4, 0.5, 0.6],
                "EIS电阻虚部": [0.7, 0.8, 0.9],
            }
        )

        features, labels = build_raw_self_window_feature_arrays(
            frame,
            group_sets=[{"g"}],
            group_to_int={"g": 0},
            group_column="__group_key__",
            stack_cols=stack_cols,
            cond_cols=cond_cols,
            label_col="类型",
            window_size=2,
            strides=[1],
        )

        self.assertEqual(labels.tolist(), [1, 1])
        self.assertEqual(features.shape, (2, 18))
        self.assertEqual(features[0, :6].tolist(), [100.0, 10.0, 1.0, 102.0, 12.0, 2.0])
        self.assertEqual(features[0, 6:9].tolist(), [55.0, 5.5, 8.5])

    def test_parse_feature_keys_accepts_plus_separated_aux_features(self) -> None:
        from scripts.plot_feature_embeddings import parse_feature_keys

        self.assertEqual(parse_feature_keys("z_op+z_eis"), ["z_op", "z_eis"])
        self.assertEqual(parse_feature_keys("h"), ["h"])

    def test_display_panel_title_wraps_long_titles(self) -> None:
        from scripts.plot_feature_embeddings import display_panel_title

        title = "(c) UniShape features before shape-aware adapter"

        self.assertIn("\n", display_panel_title(title, width=32))

    def test_compact_panel_label_keeps_only_subplot_marker(self) -> None:
        from scripts.plot_feature_embeddings import compact_panel_label

        self.assertEqual(compact_panel_label("(a) Raw input features"), "(a)")
        self.assertEqual(compact_panel_label("(d) UniShape final features after shape-aware adapter"), "(d)")

    def test_panel_caption_text_lists_full_meanings(self) -> None:
        from scripts.plot_feature_embeddings import panel_caption_text

        captions = panel_caption_text(
            [
                "(a) Raw input features",
                "(b) Baseline model features",
                "(c) UniShape features before shape-aware adapter",
                "(d) UniShape final features after shape-aware adapter",
            ]
        )

        self.assertIn("(a) Raw input features", captions)
        self.assertIn("(d) UniShape final features after shape-aware adapter", captions)
        self.assertIn("\n", captions)

    def test_baseline_shape_dataset_uses_npz_shapes_without_train_split(self) -> None:
        from scripts.plot_feature_embeddings import baseline_shape_dataset

        data = {
            "x_op": np.zeros((3, 2, 8), dtype=np.float32),
            "x_eis": np.zeros((3, 4, 6), dtype=np.float32),
            "x_cond": np.zeros((3, 5), dtype=np.float32),
        }

        shape_ds = baseline_shape_dataset(data)

        self.assertEqual(tuple(shape_ds.x_op.shape), (1, 2, 8))
        self.assertEqual(tuple(shape_ds.x_eis.shape), (1, 4, 6))
        self.assertEqual(tuple(shape_ds.x_cond.shape), (1, 5))

    def test_extract_torch_baseline_features_uses_penultimate_mlp_layer(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this Python environment")
        import torch

        from scripts.plot_feature_embeddings import extract_torch_baseline_features
        from scripts.run_official_baseline_experiments import MLPBaseline

        model = MLPBaseline(input_dim=7, hidden_dim=5, num_classes=3, dropout=0.0)
        model.eval()
        x_op = torch.ones(4, 1, 3)
        x_eis = torch.ones(4, 1, 2)
        x_cond = torch.ones(4, 2)

        features = extract_torch_baseline_features(model, "mlp", x_op, x_eis, x_cond)

        self.assertEqual(tuple(features.shape), (4, 5))


if __name__ == "__main__":
    unittest.main()
