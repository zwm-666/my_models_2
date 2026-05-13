"""Generate public-dataset interpretability figures for CAPT-UniShape."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import FuelCellNPZDataset, build_model_from_config, load_config, sync_config_with_dataset


def summarize_channel_attention(channel_weights: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    weights = channel_weights.detach().cpu()
    y = labels.detach().cpu()
    class_ids = sorted(int(v) for v in torch.unique(y).tolist())
    rows: list[torch.Tensor] = []
    for class_id in class_ids:
        mask = y == int(class_id)
        rows.append(weights[mask].mean(dim=0))
    matrix = torch.stack(rows, dim=0)
    return {"class_ids": class_ids, "matrix": matrix.numpy()}


def _collect_attention_outputs(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    op_weights: list[torch.Tensor] = []
    eis_weights: list[torch.Tensor] = []
    op_temporal: list[torch.Tensor] = []
    eis_temporal: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x_op, x_eis, x_cond, labels in loader:
            _, aux = model(x_op.to(device), x_eis.to(device), x_cond.to(device))
            op_weights.append(aux["op_channel_weights"].detach().cpu())
            eis_weights.append(aux["eis_channel_weights"].detach().cpu())
            op_temporal.append(aux["op_temporal_attention"].detach().cpu())
            eis_temporal.append(aux["eis_temporal_attention"].detach().cpu())
            labels_all.append(labels.detach().cpu())
    return {
        "labels": torch.cat(labels_all, dim=0),
        "op_channel_weights": torch.cat(op_weights, dim=0),
        "eis_channel_weights": torch.cat(eis_weights, dim=0),
        "op_temporal_attention": torch.cat(op_temporal, dim=0),
        "eis_temporal_attention": torch.cat(eis_temporal, dim=0),
    }


def _plot_heatmap(matrix: np.ndarray, x_labels: list[str], y_labels: list[str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 0.8), max(3.5, len(y_labels) * 0.8)))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _mean_temporal_attention(temporal_attention: torch.Tensor) -> np.ndarray:
    # [N, C, T_tokens] -> [C, T_tokens]
    return temporal_attention.mean(dim=0).numpy()


def run_attention_plot(
    config_path: Path,
    data_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    split: str = "test",
) -> dict[str, Path]:
    dataset_all = FuelCellNPZDataset(data_path)
    data = np.load(data_path)
    split_map = {"train": 0, "val": 1, "test": 2}
    indices = np.where(np.asarray(data["split"], dtype=np.int64) == split_map[split])[0]
    dataset = FuelCellNPZDataset(data_path, indices)
    config = sync_config_with_dataset(load_config(config_path), dataset_all)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    batch_size = int(config.get("experiment", {}).get("batch_size", 8))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = _collect_attention_outputs(model, loader, device)

    op_summary = summarize_channel_attention(outputs["op_channel_weights"], outputs["labels"])
    eis_summary = summarize_channel_attention(outputs["eis_channel_weights"], outputs["labels"])
    class_labels = [f"class_{cid}" for cid in op_summary["class_ids"]]
    op_channel_labels = ["stack_voltage", "stack_current", "stack_power"][: op_summary["matrix"].shape[1]]
    eis_channel_labels = ["response", "gradient", "cumulative", "freq_axis"][: eis_summary["matrix"].shape[1]]

    output_dir.mkdir(parents=True, exist_ok=True)
    op_path = output_dir / "public_ac_voltage_op_channel_attention.png"
    eis_path = output_dir / "public_ac_voltage_eis_channel_attention.png"
    op_temporal_path = output_dir / "public_ac_voltage_op_temporal_attention.png"
    eis_temporal_path = output_dir / "public_ac_voltage_eis_temporal_attention.png"

    _plot_heatmap(op_summary["matrix"], op_channel_labels, class_labels, "Public AC Voltage OP Channel Attention", op_path)
    _plot_heatmap(eis_summary["matrix"], eis_channel_labels, class_labels, "Public AC Voltage EIS Channel Attention", eis_path)
    _plot_heatmap(_mean_temporal_attention(outputs["op_temporal_attention"]), [f"t{i}" for i in range(outputs["op_temporal_attention"].shape[-1])], op_channel_labels, "Public AC Voltage OP Temporal Attention", op_temporal_path)
    _plot_heatmap(_mean_temporal_attention(outputs["eis_temporal_attention"]), [f"t{i}" for i in range(outputs["eis_temporal_attention"].shape[-1])], eis_channel_labels, "Public AC Voltage EIS Temporal Attention", eis_temporal_path)

    return {
        "op_channel": op_path,
        "eis_channel": eis_path,
        "op_temporal": op_temporal_path,
        "eis_temporal": eis_temporal_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制公开数据集 CAPT-UniShape 注意力解释图")
    parser.add_argument("--config", default="configs/rbf_kanfusion.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/public_interpretability")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_attention_plot(
        config_path=ROOT / str(args.config),
        data_path=ROOT / str(args.data),
        checkpoint_path=ROOT / str(args.checkpoint),
        output_dir=ROOT / str(args.output_dir),
        split=str(args.split),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
