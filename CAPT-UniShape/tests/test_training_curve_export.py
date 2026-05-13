from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch


class TrainingCurveExportTests(unittest.TestCase):
    def test_export_history_rows_preserves_epoch_metrics_and_lr(self) -> None:
        from scripts.export_training_curves import export_history_rows

        history = [
            {"epoch": 1.0, "train_loss": 0.912345, "train_accuracy": 0.6, "train_macro_f1": 0.55, "val_accuracy": 0.7, "val_macro_f1": 0.68, "lr": 1e-3},
            {"epoch": 2.0, "train_loss": 0.5, "train_accuracy": 0.8, "train_macro_f1": 0.79, "val_accuracy": 0.85, "val_macro_f1": 0.84, "lr": 8e-4},
        ]

        rows = export_history_rows(history)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["epoch"], 1)
        self.assertEqual(rows[0]["train_loss"], 0.9123)
        self.assertAlmostEqual(rows[1]["train_loss"], 0.5)
        self.assertAlmostEqual(rows[1]["val_macro_f1"], 0.84)
        self.assertAlmostEqual(rows[1]["lr"], 8e-4)

    def test_load_history_from_checkpoint_reads_embedded_history(self) -> None:
        from scripts.export_training_curves import load_history_from_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "best.ckpt"
            torch.save({"history": [{"epoch": 1.0, "train_loss": 1.2}]}, ckpt)

            history = load_history_from_checkpoint(ckpt)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["epoch"], 1.0)


if __name__ == "__main__":
    unittest.main()
