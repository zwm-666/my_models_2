"""Create EIS/impedance shape-analysis figures and descriptor tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["正常", "过湿", "过干"]
CLASS_DISPLAY_NAMES = ["Normal", "Over-wet", "Over-dry"]
SPLIT_TO_VALUE = {"train": 0, "val": 1, "test": 2}
CHANNEL_NAMES = [
    "Interpolated impedance-statistic curve",
    "First-difference shape",
    "Centered cumulative shape",
    "Normalized coordinate",
]
PANEL_LABEL_KWARGS = {"x": 0.02, "y": 1.08, "fontweight": "normal"}
DESCRIPTOR_FIELDS = [
    "split",
    "label",
    "class_name",
    "curve_mean",
    "curve_std",
    "curve_range",
    "gradient_energy",
    "gradient_abs_mean",
    "cumulative_range",
    "peak_position",
    "trough_position",
]


def _as_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def split_indices(data: Any, split: str) -> np.ndarray[Any, np.dtype[np.int64]]:
    normalized = str(split).lower()
    labels = np.asarray(data["labels"], dtype=np.int64)
    if normalized == "all":
        return np.arange(labels.shape[0], dtype=np.int64)
    if normalized not in SPLIT_TO_VALUE:
        raise ValueError(f"Unsupported split={split!r}. Use train, val, test or all.")
    if "split" not in data:
        raise ValueError("NPZ data must contain split for named split selection.")
    split_values = np.asarray(data["split"], dtype=np.int64)
    return np.where(split_values == SPLIT_TO_VALUE[normalized])[0].astype(np.int64)


def classwise_mean_std(
    values: np.ndarray[Any, Any],
    labels: np.ndarray[Any, np.dtype[np.int64]],
) -> dict[int, dict[str, Any]]:
    x = np.asarray(values, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("values and labels must have the same first dimension.")
    stats: dict[int, dict[str, Any]] = {}
    for label in sorted(np.unique(y).tolist()):
        subset = x[y == int(label)]
        stats[int(label)] = {
            "mean": subset.mean(axis=0),
            "std": subset.std(axis=0),
            "count": int(subset.shape[0]),
        }
    return stats


def shape_descriptors(
    x_eis: np.ndarray[Any, Any],
    labels: np.ndarray[Any, np.dtype[np.int64]],
    split_name: str,
) -> list[dict[str, Any]]:
    arr = np.asarray(x_eis, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if arr.ndim != 3 or arr.shape[1] < 3:
        raise ValueError("x_eis must have shape [N, at least 3 channels, L].")
    if arr.shape[0] != y.shape[0]:
        raise ValueError("x_eis and labels must have the same first dimension.")
    positions = np.linspace(0.0, 1.0, arr.shape[2], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for sample, label in zip(arr, y):
        curve = sample[0]
        gradient = sample[1]
        cumulative = sample[2]
        rows.append(
            {
                "split": split_name,
                "label": int(label),
                "class_name": CLASS_NAMES[int(label)] if 0 <= int(label) < len(CLASS_NAMES) else str(int(label)),
                "curve_mean": float(np.mean(curve)),
                "curve_std": float(np.std(curve)),
                "curve_range": float(np.max(curve) - np.min(curve)),
                "gradient_energy": float(np.sum(np.square(gradient))),
                "gradient_abs_mean": float(np.mean(np.abs(gradient))),
                "cumulative_range": float(np.max(cumulative) - np.min(cumulative)),
                "peak_position": float(positions[int(np.argmax(curve))]),
                "trough_position": float(positions[int(np.argmin(curve))]),
            }
        )
    return rows


def descriptor_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [field for field in DESCRIPTOR_FIELDS if field not in {"split", "label", "class_name"}]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["label"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for label in sorted(grouped):
        group_rows = grouped[label]
        out: dict[str, Any] = {
            "split": group_rows[0]["split"],
            "label": label,
            "class_name": group_rows[0]["class_name"],
            "n": len(group_rows),
        }
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in group_rows], dtype=np.float32)
            out[f"{field}_mean"] = float(values.mean())
            out[f"{field}_std"] = float(values.std())
        summary.append(out)
    return summary


def _format_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return value


def write_descriptor_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(DESCRIPTOR_FIELDS)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row.get(field, "")) for field in fields})


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["split", "label", "class_name", "n"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row.get(field, "")) for field in fields})


def centroid_distance_matrix(x_eis: np.ndarray[Any, Any], labels: np.ndarray[Any, np.dtype[np.int64]]) -> tuple[np.ndarray[Any, Any], list[int]]:
    flat = np.asarray(x_eis, dtype=np.float32).reshape(x_eis.shape[0], -1)
    y = np.asarray(labels, dtype=np.int64)
    stats = classwise_mean_std(flat, y)
    label_order = sorted(stats)
    centroids = np.stack([stats[label]["mean"] for label in label_order], axis=0)
    distance = np.zeros((len(label_order), len(label_order)), dtype=np.float32)
    for i, left in enumerate(centroids):
        for j, right in enumerate(centroids):
            distance[i, j] = float(np.linalg.norm(left - right))
    return distance, label_order


def plot_shape_analysis(
    x_eis: np.ndarray[Any, Any],
    labels: np.ndarray[Any, np.dtype[np.int64]],
    output_path: Path,
    *,
    split_name: str,
    source_note: str,
) -> None:
    import matplotlib.pyplot as plt
    from outputs.paper_figures.plot_style import (
        LEGEND_KWARGS,
        TOP_LEGEND_ANCHOR_Y,
        add_panel_tag,
        apply_paper_style,
        save_paper_figure,
        style_axes,
        style_legend_frame,
    )

    apply_paper_style()
    colors = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}
    label_names = {idx: name for idx, name in enumerate(CLASS_DISPLAY_NAMES)}
    x_axis = np.linspace(0.0, 1.0, x_eis.shape[2], dtype=np.float32)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8))
    for ax, channel, panel in zip(axes.ravel(), [0, 1, 2, 3], ["(a)", "(b)", "(c)", "(d)"]):
        stats = classwise_mean_std(x_eis[:, channel, :], labels)
        for label, item in stats.items():
            mean = item["mean"]
            std = item["std"]
            color = colors.get(label, "#777777")
            ax.plot(x_axis, mean, color=color, linewidth=1.6, label=f"{label_names.get(label, label)} (n={item['count']})")
            ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        add_panel_tag(ax, panel)
        ax.set_xlabel("Normalized statistic coordinate")
        ax.set_ylabel(CHANNEL_NAMES[channel])
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.margins(x=0)
        style_axes(ax, grid_axis="both")
    handles, labels_text = axes.ravel()[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels_text,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, TOP_LEGEND_ANCHOR_Y),
        handlelength=1.8,
        columnspacing=1.6,
        **LEGEND_KWARGS,
    )
    style_legend_frame(legend)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.80, bottom=0.12, wspace=0.22, hspace=0.48)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, str(output_path.with_suffix("")), formats=("svg", "png"))
    plt.close(fig)


def write_meta(
    path: Path,
    *,
    npz_path: Path,
    output_figure: Path,
    descriptor_csv: Path,
    summary_csv: Path,
    split: str,
    indices: np.ndarray[Any, np.dtype[np.int64]],
    labels: np.ndarray[Any, np.dtype[np.int64]],
) -> None:
    counts = {CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else str(int(label)): int((labels == label).sum()) for label in sorted(np.unique(labels).tolist())}
    meta = {
        "analysis": "constructed EIS/impedance statistical shape analysis",
        "source_npz": str(npz_path),
        "split": split,
        "sample_count": int(indices.shape[0]),
        "class_counts": counts,
        "label_mapping": {"正常": 0, "过湿/水淹": 1, "过干/膜干": 2},
        "x_eis_channels": CHANNEL_NAMES,
        "note": "x_eis is constructed from nine impedance/EIS statistical features; it is not raw full-frequency EIS spectra.",
        "figure": str(output_figure),
        "descriptor_csv": str(descriptor_csv),
        "summary_csv": str(summary_csv),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        default="data/processed/supplemental_fault_features_seed44_8_2/official_self_stack_impedance_eis_w64_8_2.npz",
        help="Input NPZ with x_eis, labels and split.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Split to analyze.")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_figures/shape_analysis",
        help="Directory for figure, CSV and metadata outputs.",
    )
    parser.add_argument("--prefix", default="supplemental_eis_shape", help="Output filename prefix.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    npz_path = _as_project_path(args.npz)
    output_dir = _as_project_path(args.output_dir)
    data = np.load(npz_path)
    indices = split_indices(data, args.split)
    labels = np.asarray(data["labels"][indices], dtype=np.int64)
    x_eis = np.asarray(data["x_eis"][indices], dtype=np.float32)
    if indices.size == 0:
        raise ValueError(f"No samples selected for split={args.split!r}.")
    descriptor_rows = shape_descriptors(x_eis, labels, args.split)
    summary_rows = descriptor_summary(descriptor_rows)
    figure_path = output_dir / f"{args.prefix}_analysis.png"
    descriptor_csv = output_dir / f"{args.prefix}_descriptors.csv"
    summary_csv = output_dir / f"{args.prefix}_class_summary.csv"
    meta_json = output_dir / f"{args.prefix}_analysis.meta.json"
    plot_shape_analysis(
        x_eis,
        labels,
        figure_path,
        split_name=args.split,
        source_note=npz_path.name,
    )
    write_descriptor_csv(descriptor_rows, descriptor_csv)
    write_summary_csv(summary_rows, summary_csv)
    write_meta(
        meta_json,
        npz_path=npz_path,
        output_figure=figure_path,
        descriptor_csv=descriptor_csv,
        summary_csv=summary_csv,
        split=args.split,
        indices=indices,
        labels=labels,
    )
    print(json.dumps({"figure": str(figure_path), "descriptor_csv": str(descriptor_csv), "summary_csv": str(summary_csv), "meta": str(meta_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
