from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from train import FuelCellNPZDataset, split_train_holdout_by_groups


class TrainHoldoutSplitTests(unittest.TestCase):
    def test_group_holdout_keeps_groups_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = Path(tmp_dir) / "toy.npz"
            labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
            group_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
            split = np.zeros_like(labels)
            x_op = np.zeros((labels.shape[0], 3, 8), dtype=np.float32)
            x_eis = np.zeros((labels.shape[0], 4, 8), dtype=np.float32)
            x_cond = np.zeros((labels.shape[0], 2), dtype=np.float32)
            np.savez_compressed(npz_path, x_op=x_op, x_eis=x_eis, x_cond=x_cond, labels=labels, group_ids=group_ids, split=split)

            dataset = FuelCellNPZDataset(npz_path)
            fit_ds, holdout_ds = split_train_holdout_by_groups(dataset, holdout_ratio=0.5, seed=44)

            self.assertIsNotNone(holdout_ds)
            fit_group_ids = {int(dataset.group_ids[idx]) for idx in fit_ds.indices}  # type: ignore[attr-defined]
            holdout_group_ids = {int(dataset.group_ids[idx]) for idx in holdout_ds.indices}  # type: ignore[attr-defined]
            self.assertTrue(fit_group_ids.isdisjoint(holdout_group_ids))


if __name__ == "__main__":
    unittest.main()
