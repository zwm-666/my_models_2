from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is required to import the experiment runner")

from experiments.run_official_baseline_experiments import _make_nested_train_subset_npzs


class NestedFixedSplitTests(unittest.TestCase):
    def test_fixed_test_train_subsets_are_nested_and_keep_heldout_fixed(self) -> None:
        with self.subTest("build nested ratio npz files"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                base_npz = tmp_path / "base.npz"
                labels = np.array(
                    [0] * 8
                    + [1] * 8
                    + [2] * 16
                    + [0, 0, 1, 1, 2, 2]
                    + [0, 0, 1, 1, 2, 2],
                    dtype=np.int64,
                )
                split = np.array([0] * 32 + [1] * 6 + [2] * 6, dtype=np.int64)
                sample_ids = np.arange(labels.shape[0], dtype=np.float32)
                np.savez_compressed(
                    base_npz,
                    x_op=sample_ids.reshape(-1, 1, 1),
                    x_eis=sample_ids.reshape(-1, 1, 1),
                    x_cond=sample_ids.reshape(-1, 1),
                    labels=labels,
                    split=split,
                )

                outputs = {
                    ratio: tmp_path / f"{ratio}.npz"
                    for ratio in ("5_5", "6_4", "7_3", "8_2")
                }

                _make_nested_train_subset_npzs(
                    base_npz,
                    outputs,
                    fixed_test_ratio="8_2",
                    seed=44,
                )

                train_sets: dict[str, set[int]] = {}
                base_val = set(np.where(split == 1)[0].tolist())
                base_test = set(np.where(split == 2)[0].tolist())
                for ratio, output_npz in outputs.items():
                    with np.load(output_npz) as data:
                        original_indices = data["x_cond"].reshape(-1).astype(int)
                        output_split = data["split"]
                        train_sets[ratio] = set(original_indices[output_split == 0].tolist())
                        val_set = set(original_indices[output_split == 1].tolist())
                        test_set = set(original_indices[output_split == 2].tolist())

                    self.assertEqual(val_set, base_val)
                    self.assertEqual(test_set, base_test)

                    summary = json.loads(output_npz.with_suffix(".summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["split_protocol"], "nested_fixed_test_train_subset")

                self.assertLess(len(train_sets["5_5"]), len(train_sets["6_4"]))
                self.assertLess(len(train_sets["6_4"]), len(train_sets["7_3"]))
                self.assertLess(len(train_sets["7_3"]), len(train_sets["8_2"]))
                self.assertTrue(train_sets["5_5"] <= train_sets["6_4"])
                self.assertTrue(train_sets["6_4"] <= train_sets["7_3"])
                self.assertTrue(train_sets["7_3"] <= train_sets["8_2"])


if __name__ == "__main__":
    unittest.main()

