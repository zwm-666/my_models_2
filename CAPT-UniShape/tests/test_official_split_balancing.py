from __future__ import annotations

import unittest

import numpy as np

from experiments.build_official_npz_from_self_excel import _split_quality


class OfficialSplitBalancingTests(unittest.TestCase):
    def setUp(self) -> None:
        # 3 classes (0/1/2), each group has enough rows for multiple windows.
        self.row_counts = {
            "g0": 96,
            "g1": 96,
            "g2": 96,
            "g3": 96,
            "g4": 96,
            "g5": 96,
            "g6": 96,
            "g7": 96,
            "g8": 96,
            "g9": 96,
            "g10": 96,
            "g11": 96,
        }
        self.group_label_map = {
            "g0": 0,
            "g1": 0,
            "g2": 0,
            "g3": 0,
            "g4": 1,
            "g5": 1,
            "g6": 1,
            "g7": 1,
            "g8": 2,
            "g9": 2,
            "g10": 2,
            "g11": 2,
        }
        self.labels = ["0", "1", "2"]

    def test_train_group_floor_can_reject_split_even_when_eval_passes(self) -> None:
        # class-0 in train only has two groups -> should fail when min-train-class-groups=3
        g_tr = np.array(["g0", "g1", "g4", "g5", "g8", "g9"], dtype=object)
        g_va = np.array(["g2", "g6", "g10"], dtype=object)
        g_te = np.array(["g3", "g7", "g11"], dtype=object)

        quality = _split_quality(
            row_counts=self.row_counts,
            group_label_map=self.group_label_map,
            g_tr=g_tr,
            g_va=g_va,
            g_te=g_te,
            window_size=64,
            stride_train=16,
            stride_val=32,
            stride_eval=32,
            labels=self.labels,
            min_eval_class_windows=1,
            min_eval_class_groups=1,
            min_train_class_windows=1,
            min_train_class_groups=3,
            min_val_class_groups=1,
            min_test_class_groups=1,
            prefer_balanced_train_groups=False,
        )

        self.assertEqual(quality["min_train_class_groups"], 2)
        self.assertFalse(bool(quality["passed"]))

    def test_prefer_balanced_train_groups_penalizes_skewed_train_groups(self) -> None:
        balanced_g_tr = np.array(["g0", "g1", "g4", "g5", "g8", "g9"], dtype=object)
        skewed_g_tr = np.array(["g0", "g1", "g4", "g5", "g6", "g8"], dtype=object)
        g_va = np.array(["g3", "g6", "g10"], dtype=object)
        g_te = np.array(["g7", "g5", "g11"], dtype=object)

        balanced = _split_quality(
            row_counts=self.row_counts,
            group_label_map=self.group_label_map,
            g_tr=balanced_g_tr,
            g_va=g_va,
            g_te=g_te,
            window_size=64,
            stride_train=16,
            stride_val=32,
            stride_eval=32,
            labels=self.labels,
            min_eval_class_windows=0,
            min_eval_class_groups=0,
            min_train_class_windows=1,
            min_train_class_groups=1,
            min_val_class_groups=1,
            min_test_class_groups=1,
            prefer_balanced_train_groups=True,
        )
        skewed = _split_quality(
            row_counts=self.row_counts,
            group_label_map=self.group_label_map,
            g_tr=skewed_g_tr,
            g_va=g_va,
            g_te=g_te,
            window_size=64,
            stride_train=16,
            stride_val=32,
            stride_eval=32,
            labels=self.labels,
            min_eval_class_windows=0,
            min_eval_class_groups=0,
            min_train_class_windows=1,
            min_train_class_groups=1,
            min_val_class_groups=1,
            min_test_class_groups=1,
            prefer_balanced_train_groups=True,
        )

        self.assertGreater(int(balanced["score"]), int(skewed["score"]))


if __name__ == "__main__":
    unittest.main()

