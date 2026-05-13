"""Export train/validation curve data and plots from saved checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_history_from_checkpoint(checkpoint_path: Path) -> list[dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = payload.get("history", [])
    if not isinstance(history, list):
        raise TypeError(f"{checkpoint_path} 中的 history 不是 list")
    return [dict(item) for item in history]


def export_history_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(history, start=1):
        rows.append(
            {
                "epoch": int(round(float(item.get("epoch", idx)))),
                "train_loss": float(item.get("train_loss", 0.0)),
                "train_accuracy": float(item.get("train_accuracy", 0.0)),
                "train_macro_f1": float(item.get("train_macro_f1", 0.0)),
                "val_accuracy": float(item.get("val_accuracy", 0.0)),
                "val_macro_f1": float(item.get("val_macro_f1", 0.0)),
                "val_class0_recall": float(item.get("val_class0_recall", 0.0)),
                "train_holdout_macro_f1": float(item.get("train_holdout_macro_f1", 0.0)),
                "train_val_gap_penalty": float(item.get("train_val_gap_penalty", 0.0)),
                "selection_score": float(item.get("selection_score", 0.0)),
                "lr": float(item.get("lr", 0.0)),
            }
        )
    return rows


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _plot_metric(rows: list[dict[str, Any]], y_keys: list[tuple[str, str]], title: str, ylabel: str, output_path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in y_keys:
        ax.plot(epochs, [row[key] for row in rows], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_training_curves(checkpoint_path: Path, output_dir: Path) -> dict[str, Path]:
    history = load_history_from_checkpoint(checkpoint_path)
    rows = export_history_rows(history)
    csv_path = output_dir / "training_curves_data.csv"
    json_path = output_dir / "training_curves_data.json"
    _write_rows_csv(csv_path, rows)
    _write_rows_json(json_path, rows)

    loss_path = output_dir / "training_loss_curve.png"
    acc_path = output_dir / "training_accuracy_curve.png"
    f1_path = output_dir / "training_f1_curve.png"
    lr_path = output_dir / "training_lr_curve.png"

    _plot_metric(rows, [("train_loss", "Train Loss")], "Training Loss Curve", "Loss", loss_path)
    _plot_metric(rows, [("train_accuracy", "Train Accuracy"), ("val_accuracy", "Validation Accuracy")], "Training/Validation Accuracy Curve", "Accuracy", acc_path)
    _plot_metric(rows, [("train_macro_f1", "Train Macro-F1"), ("val_macro_f1", "Validation Macro-F1")], "Training/Validation F1 Curve", "Macro-F1", f1_path)
    _plot_metric(rows, [("lr", "Learning Rate")], "Learning Rate Curve", "LR", lr_path)

    return {"csv": csv_path, "json": json_path, "loss": loss_path, "accuracy": acc_path, "f1": f1_path, "lr": lr_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出训练/验证曲线数据与图片")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = export_training_curves(Path(args.checkpoint), Path(args.output_dir))
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
