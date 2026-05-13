from __future__ import annotations

import unittest

import numpy as np


class SingleSeedFoldTests(unittest.TestCase):
    def test_stratified_group_folds_use_one_seed_and_disjoint_tests(self) -> None:
        from scripts.run_official_single_seed_fold_experiments import _make_stratified_group_folds

        groups = np.array([f"g{i:02d}" for i in range(15)], dtype=object)
        labels = np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=np.int64)

        folds = _make_stratified_group_folds(groups, labels, n_folds=5, seed=44, val_fraction=0.25)

        self.assertEqual(len(folds), 5)
        all_test_groups: list[object] = []
        for fold in folds:
            train = set(fold["train"].tolist())
            val = set(fold["val"].tolist())
            test = set(fold["test"].tolist())
            self.assertFalse(train & val)
            self.assertFalse(train & test)
            self.assertFalse(val & test)
            self.assertEqual(len(test), 3)
            test_labels = labels[[np.where(groups == group)[0][0] for group in fold["test"]]]
            self.assertEqual(set(test_labels.tolist()), {0, 1, 2})
            all_test_groups.extend(fold["test"].tolist())

        self.assertEqual(set(all_test_groups), set(groups.tolist()))
        self.assertEqual(len(all_test_groups), len(set(all_test_groups)))


if __name__ == "__main__":
    unittest.main()
