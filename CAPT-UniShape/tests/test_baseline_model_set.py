from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import run_official_baseline_experiments as baseline


class BaselineModelSetTests(unittest.TestCase):
    def _toy_dataset(self) -> baseline.BaselineNPZDataset:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        npz_path = Path(tmpdir.name) / "toy.npz"
        np.savez_compressed(
            npz_path,
            x_op=np.ones((6, 2, 8), dtype=np.float32),
            x_eis=np.ones((6, 1, 6), dtype=np.float32),
            x_cond=np.ones((6, 3), dtype=np.float32),
            split=np.asarray([0, 0, 0, 1, 2, 2], dtype=np.int64),
            labels=np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        )
        return baseline.BaselineNPZDataset(npz_path, split_value=0)

    def test_requested_baseline_models_are_registered(self) -> None:
        expected = {
            "xgboost",
            "lightgbm",
            "mlp",
            "tcn",
            "autoformer",
        }

        self.assertTrue(expected.issubset(set(baseline.MODEL_CATEGORIES)))

    def test_new_torch_baselines_build_and_forward(self) -> None:
        import torch

        train_ds = self._toy_dataset()
        for model_key in ("tcn", "autoformer"):
            model = baseline._build_torch_model(
                model_key,
                train_ds,
                num_classes=3,
                hidden_dim=8,
                d_model=8,
                num_layers=1,
                dropout=0.0,
            )
            logits = model(train_ds.x_op, train_ds.x_eis, train_ds.x_cond)

            self.assertEqual(tuple(logits.shape), (3, 3))
            self.assertTrue(torch.isfinite(logits).all())

    def test_boosting_models_are_real_optional_dependencies(self) -> None:
        for model_key, package_name in (("xgboost", "xgboost"), ("lightgbm", "lightgbm")):
            if importlib.util.find_spec(package_name) is None:
                with self.assertRaisesRegex(ImportError, package_name):
                    baseline._build_ml_model(model_key, seed=44, rf_estimators=200)
            else:
                model = baseline._build_ml_model(model_key, seed=44, rf_estimators=200)
                self.assertIn(package_name.lower(), type(model).__module__.lower())


if __name__ == "__main__":
    unittest.main()
