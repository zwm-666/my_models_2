"""Estimate constructed EIS/impedance shape-channel importance by perturbation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["正常", "过湿", "过干"]
CLASS_DISPLAY_NAMES = ["Normal", "Over-wet", "Over-dry"]
CHANNEL_NAMES = [
    "Interpolated curve",
    "Gradient shape",
    "Cumulative shape",
    "Coordinate axis",
]
SPLIT_TO_VALUE = {"train": 0, "val": 1, "test": 2}
IMPORTANCE_FIELDS = [
    "method",
    "channel",
    "channel_name",
    "accuracy",
    "macro_f1",
    "mean_true_probability",
    "accuracy_drop",
    "macro_f1_drop",
    "mean_true_probability_drop",
]


def _as_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _split_indices(split: np.ndarray[Any, Any], name: str) -> np.ndarray[Any, np.dtype[np.int64]]:
    normalized = str(name).lower()
    if normalized not in SPLIT_TO_VALUE:
        raise ValueError(f"Unsupported split={name!r}. Use train, val or test.")
    return np.where(np.asarray(split, dtype=np.int64) == SPLIT_TO_VALUE[normalized])[0].astype(np.int64)


def flatten_shape(x_eis: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    arr = np.asarray(x_eis, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("x_eis must have shape [N, C, L].")
    return arr.reshape(arr.shape[0], -1)


def replace_channel_with_reference(
    x_eis: np.ndarray[Any, Any],
    *,
    channel: int,
    reference: np.ndarray[Any, Any],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    arr = np.asarray(x_eis, dtype=np.float32).copy()
    ref = np.asarray(reference, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("x_eis must have shape [N, C, L].")
    if ref.shape != arr.shape[1:]:
        raise ValueError(f"reference must have shape {arr.shape[1:]}, got {ref.shape}.")
    arr[:, int(channel), :] = ref[int(channel), :]
    return arr


def evaluate_shape_model(model: Any, x_eis: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    features = flatten_shape(x_eis)
    pred = model.predict(features)
    prob = model.predict_proba(features)
    true_prob = prob[np.arange(y.shape[0]), y]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mean_true_probability": float(np.mean(true_prob)),
    }


def metric_drop_row(
    *,
    channel: int,
    channel_name: str,
    baseline: dict[str, float],
    perturbed: dict[str, float],
    method: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "channel": int(channel),
        "channel_name": channel_name,
        "accuracy": float(perturbed["accuracy"]),
        "macro_f1": float(perturbed["macro_f1"]),
        "mean_true_probability": float(perturbed["mean_true_probability"]),
        "accuracy_drop": float(baseline["accuracy"] - perturbed["accuracy"]),
        "macro_f1_drop": float(baseline["macro_f1"] - perturbed["macro_f1"]),
        "mean_true_probability_drop": float(baseline["mean_true_probability"] - perturbed["mean_true_probability"]),
    }


def compute_shape_importance(
    train_x_eis: np.ndarray[Any, Any],
    train_labels: np.ndarray[Any, Any],
    test_x_eis: np.ndarray[Any, Any],
    test_labels: np.ndarray[Any, Any],
    *,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=int(seed),
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(flatten_shape(train_x_eis), np.asarray(train_labels, dtype=np.int64))
    baseline = evaluate_shape_model(model, test_x_eis, test_labels)
    reference = np.asarray(train_x_eis, dtype=np.float32).mean(axis=0)
    rows: list[dict[str, Any]] = []
    for channel, name in enumerate(CHANNEL_NAMES[: test_x_eis.shape[1]]):
        replaced = replace_channel_with_reference(test_x_eis, channel=channel, reference=reference)
        perturbed = evaluate_shape_model(model, replaced, test_labels)
        rows.append(
            metric_drop_row(
                channel=channel,
                channel_name=name,
                baseline=baseline,
                perturbed=perturbed,
                method="train_mean_replacement",
            )
        )
    return baseline, rows


def _format_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return value


def write_importance_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMPORTANCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row.get(field, "")) for field in IMPORTANCE_FIELDS})


def plot_shape_importance(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    baseline: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: float(row["mean_true_probability_drop"]), reverse=True)
    labels = [str(row["channel_name"]) for row in ordered]
    values = np.asarray([float(row["mean_true_probability_drop"]) for row in ordered], dtype=np.float32)
    f1_drop = np.asarray([float(row["macro_f1_drop"]) for row in ordered], dtype=np.float32)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color="#4C78A8", width=0.62, label="True-class probability drop")
    ax.plot(x, f1_drop, color="#F58518", marker="o", linewidth=1.4, label="Macro-F1 drop")
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Performance drop after channel replacement")
    ax.text(0.02, 1.08, "(a)", transform=ax.transAxes, fontweight="normal")
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value), f"{float(value):.3f}", ha="center", va="bottom", fontsize=7)
    ax.legend(frameon=False, loc="upper right")
    fig.text(
        0.01,
        -0.04,
        f"Shape-only RandomForest baseline: Acc={baseline['accuracy']:.4f}, Macro-F1={baseline['macro_f1']:.4f}. Higher drop means higher channel importance.",
        fontsize=7,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_meta(
    path: Path,
    *,
    npz_path: Path,
    figure_path: Path,
    csv_path: Path,
    baseline: dict[str, float],
    train_count: int,
    test_count: int,
) -> None:
    meta = {
        "analysis": "shape-channel perturbation importance",
        "source_npz": str(npz_path),
        "estimator": "RandomForestClassifier shape-only proxy",
        "perturbation": "replace one x_eis channel with its train-set mean channel",
        "baseline_metrics": {key: round(float(value), 4) for key, value in baseline.items()},
        "train_count": int(train_count),
        "test_count": int(test_count),
        "channels": CHANNEL_NAMES,
        "label_mapping": {"正常": 0, "过湿/水淹": 1, "过干/膜干": 2},
        "limitation": "This is input shape-channel importance from a reproducible shape-only proxy model because no trained CAPT-UniShape checkpoint is present in the workspace.",
        "figure": str(figure_path),
        "csv": str(csv_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        default="data/processed/supplemental_fault_features_seed44_8_2/official_self_stack_impedance_eis_w64_8_2.npz",
    )
    parser.add_argument("--output-dir", default="outputs/paper_figures/shape_analysis")
    parser.add_argument("--prefix", default="supplemental_eis_shape_importance")
    parser.add_argument("--seed", type=int, default=44)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    npz_path = _as_project_path(args.npz)
    output_dir = _as_project_path(args.output_dir)
    data = np.load(npz_path)
    train_indices = _split_indices(data["split"], "train")
    test_indices = _split_indices(data["split"], "test")
    labels = np.asarray(data["labels"], dtype=np.int64)
    x_eis = np.asarray(data["x_eis"], dtype=np.float32)
    baseline, rows = compute_shape_importance(
        x_eis[train_indices],
        labels[train_indices],
        x_eis[test_indices],
        labels[test_indices],
        seed=int(args.seed),
    )
    figure_path = output_dir / f"{args.prefix}.png"
    csv_path = output_dir / f"{args.prefix}.csv"
    meta_path = output_dir / f"{args.prefix}.meta.json"
    plot_shape_importance(rows, figure_path, baseline=baseline)
    write_importance_csv(rows, csv_path)
    write_meta(
        meta_path,
        npz_path=npz_path,
        figure_path=figure_path,
        csv_path=csv_path,
        baseline=baseline,
        train_count=int(train_indices.shape[0]),
        test_count=int(test_indices.shape[0]),
    )
    print(json.dumps({"figure": str(figure_path), "csv": str(csv_path), "meta": str(meta_path), "baseline": baseline}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
