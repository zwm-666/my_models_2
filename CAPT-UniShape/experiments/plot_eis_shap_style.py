"""Create SHAP EIS feature-importance figures for the 6:4 split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STYLE_DIR = ROOT / "outputs" / "paper_figures"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from experiments.build_official_npz_from_self_excel import (  # noqa: E402
    EIS_COLS,
    LABEL_COL,
    TIME_COL,
    _choose_group_split,
    _derive_group_keys,
    _derive_segment_group_keys,
    normalize_columns,
)
from plot_style import (  # noqa: E402
    CM,
    FONT_SIZE,
    LEGEND_KWARGS,
    MODEL_COLORS,
    PANEL_TAG_POSITION,
    apply_paper_style,
    style_axes,
    style_legend_frame,
)


CLASS_ORDER = [2, 0, 1]
CLASS_COLORS = {2: MODEL_COLORS[3], 0: MODEL_COLORS[0], 1: MODEL_COLORS[2]}
CLASS_NAMES = {0: "Class 0", 1: "Class 1", 2: "Class 2"}
LABEL_ALIASES = [LABEL_COL, "类型", "label", "Label", "标签"]
FEATURE_NAME_EN = {
    "总阻抗": "Total impedance",
    "平均阻抗": "Mean impedance",
    "最高阻抗": "Maximum impedance",
    "次高阻抗": "Second-highest impedance",
    "最低阻抗": "Minimum impedance",
    "次低阻抗": "Second-lowest impedance",
    "标准差": "Standard deviation",
    "EIS电阻实部": "EIS resistance real",
    "EIS电阻虚部": "EIS resistance imaginary",
}


def _as_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _label_column(frame: pd.DataFrame) -> str:
    for col in LABEL_ALIASES:
        if col in frame.columns:
            return str(col)
    raise ValueError(f"Missing label column. Expected one of {LABEL_ALIASES}")


def english_feature_names(feature_names: list[str]) -> list[str]:
    return [FEATURE_NAME_EN.get(str(name), str(name)) for name in feature_names]


def _read_excel_with_labels(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    source = _label_column(frame)
    if source != LABEL_COL:
        frame[LABEL_COL] = frame[source]
    if frame[LABEL_COL].dtype == object:
        mapping = {"正常": 0, "过湿": 1, "水淹": 1, "过干": 2, "膜干": 2}
        values = frame[LABEL_COL].astype(str).str.strip()
        unknown = set(values.unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unknown string labels: {sorted(unknown)}")
        frame[LABEL_COL] = values.map(mapping).astype(int)
    else:
        frame[LABEL_COL] = frame[LABEL_COL].astype(int)
    return frame


def build_eis_window_features(
    *,
    excel_path: Path,
    sheet_name: str,
    window_size: int,
    stride_train: int,
    stride_val: int,
    stride_eval: int,
    split_mode: str,
    segment_gap_seconds: float,
    segment_block_seconds: float,
    segment_label_boundary: bool,
    random_state: int,
    test_size: float,
    val_size: float,
    group_split_strategy: str,
    split_retries: int,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]], dict[str, Any]]:
    frame = _read_excel_with_labels(excel_path, sheet_name)
    missing = set(EIS_COLS + [TIME_COL, LABEL_COL]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns in Excel: {sorted(missing)}")
    frame = frame.copy()
    if split_mode == "segment":
        frame["__group_key__"] = _derive_segment_group_keys(
            frame,
            gap_seconds=float(segment_gap_seconds),
            block_seconds=float(segment_block_seconds),
            label_boundary=bool(segment_label_boundary),
        )
    else:
        frame["__group_key__"] = _derive_group_keys(pd.Series(frame[TIME_COL].to_numpy()), split_mode)

    groups = np.asarray(frame["__group_key__"].unique())
    group_label_map = frame.groupby("__group_key__")[LABEL_COL].first()
    group_labels = np.array([group_label_map[group] for group in groups])
    row_counts = {group: int(count) for group, count in frame.groupby("__group_key__").size().items()}
    train_groups, val_groups, test_groups, split_quality = _choose_group_split(
        groups=groups,
        group_labels=group_labels,
        row_counts=row_counts,
        group_label_map=group_label_map,
        test_size=float(test_size),
        val_size=float(val_size),
        random_state=int(random_state),
        group_split_strategy=group_split_strategy,
        window_size=int(window_size),
        stride_train=int(stride_train),
        stride_val=int(stride_val),
        stride_eval=int(stride_eval),
        split_retries=int(split_retries),
        min_eval_class_windows=5,
        min_eval_class_groups=1,
        min_train_class_windows=1,
        min_train_class_groups=1,
        min_val_class_groups=1,
        min_test_class_groups=1,
        prefer_balanced_train_groups=False,
    )
    split_group_sets = [set(train_groups.tolist()), set(val_groups.tolist()), set(test_groups.tolist())]
    frame[EIS_COLS] = frame[EIS_COLS].fillna(0)
    train_frame = frame[frame["__group_key__"].isin(list(split_group_sets[0]))].copy()
    _, means, stds = normalize_columns(train_frame, EIS_COLS)
    for col in EIS_COLS:
        std = stds.get(col, 1.0) or 1.0
        frame[col] = (frame[col] - means.get(col, 0.0)) / std

    features: list[np.ndarray[Any, Any]] = []
    labels: list[int] = []
    split: list[int] = []
    strides = [int(stride_train), int(stride_val), int(stride_eval)]
    for split_value, group_set in enumerate(split_group_sets):
        stride = strides[split_value]
        for group in sorted(group_set, key=str):
            group_frame = frame.loc[frame["__group_key__"] == group].sort_index()
            label = int(group_frame[LABEL_COL].iloc[0])
            values = group_frame[EIS_COLS].to_numpy(dtype=np.float32)
            row_count = int(values.shape[0])
            if row_count < int(window_size):
                starts = [0]
            else:
                starts = list(range(0, row_count - int(window_size) + 1, max(1, stride)))
            for start in starts:
                window = values if row_count < int(window_size) else values[start : start + int(window_size)]
                features.append(window.mean(axis=0).astype(np.float32))
                labels.append(label)
                split.append(split_value)

    meta = {
        "feature_names": list(EIS_COLS),
        "split_quality": split_quality,
        "normalization": "z-score using 6:4 training groups",
    }
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(split, dtype=np.int64),
        meta,
    )


def classwise_mean_abs_impact(impacts: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    arr = np.asarray(impacts, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("impacts must have shape [N, F, C].")
    return np.mean(np.abs(arr), axis=0).astype(np.float32)


def top_feature_indices(classwise_matrix: np.ndarray[Any, Any], top_k: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    matrix = np.asarray(classwise_matrix, dtype=np.float32)
    totals = matrix.sum(axis=1)
    return np.argsort(-totals)[: int(top_k)].astype(np.int64)


def train_eis_model(
    features: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    split: np.ndarray[Any, Any],
    *,
    seed: int,
) -> tuple[RandomForestClassifier, dict[str, float]]:
    train_mask = np.asarray(split, dtype=np.int64) == 0
    test_mask = np.asarray(split, dtype=np.int64) == 2
    model = RandomForestClassifier(
        n_estimators=600,
        random_state=int(seed),
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(features[train_mask], labels[train_mask])
    pred = model.predict(features[test_mask])
    metrics = {
        "test_accuracy": float(accuracy_score(labels[test_mask], pred)),
        "test_macro_f1": float(f1_score(labels[test_mask], pred, average="macro", zero_division=0)),
        "train_count": int(train_mask.sum()),
        "test_count": int(test_mask.sum()),
    }
    return model, metrics


def normalize_shap_values(values: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return SHAP values as [N, F, C] for multiclass outputs."""
    if isinstance(values, list):
        return np.stack([np.asarray(item, dtype=np.float32) for item in values], axis=-1).astype(np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 2:
        return arr[:, :, None].astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected SHAP values with 2 or 3 dimensions, got {arr.shape}.")
    return arr.astype(np.float32)


def normalize_shap_interactions(values: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return SHAP interaction values as [N, F, F, C]."""
    if isinstance(values, list):
        return np.stack([np.asarray(item, dtype=np.float32) for item in values], axis=-1).astype(np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3:
        return arr[:, :, :, None].astype(np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected SHAP interaction values with 3 or 4 dimensions, got {arr.shape}.")
    return arr.astype(np.float32)


def tree_shap_values(
    model: RandomForestClassifier,
    test_features: np.ndarray[Any, Any],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    import shap

    explainer = shap.TreeExplainer(model)
    return normalize_shap_values(explainer.shap_values(test_features))


def true_class_interaction_values(
    model: RandomForestClassifier,
    test_features: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    feature_indices: np.ndarray[Any, np.dtype[np.int64]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    import shap

    explainer = shap.TreeExplainer(model)
    interactions_all = normalize_shap_interactions(explainer.shap_interaction_values(test_features))
    y = np.asarray(labels, dtype=np.int64)
    selected = np.asarray(feature_indices, dtype=np.int64)
    values = interactions_all[:, selected, :, :]
    values = values[:, :, selected, :]
    class_positions = {int(cls): idx for idx, cls in enumerate(model.classes_.tolist())}
    out = np.zeros((test_features.shape[0], len(selected), len(selected)), dtype=np.float32)
    for sample_idx, label in enumerate(y.tolist()):
        out[sample_idx] = values[sample_idx, :, :, class_positions[int(label)]]
    return out


def _setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimSun", "SimHei", "Arial", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def _add_plain_panel_tag(ax: Any, tag: str) -> None:
    ax.text(
        *PANEL_TAG_POSITION,
        tag,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
        fontweight="normal",
    )


def plot_bar(
    classwise_matrix: np.ndarray[Any, Any],
    feature_names: list[str],
    output_path: Path,
    *,
    top_k: int,
) -> None:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    order = top_feature_indices(classwise_matrix, top_k=top_k)
    ordered_names = [feature_names[int(idx)] for idx in order]
    ordered_matrix = np.asarray(classwise_matrix, dtype=np.float32)[order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(18 * CM, 11 * CM), constrained_layout=True)
    left = np.zeros(len(order), dtype=np.float32)
    for class_id in CLASS_ORDER:
        values = ordered_matrix[:, int(class_id)]
        ax.barh(
            y,
            values,
            left=left,
            color=CLASS_COLORS[int(class_id)],
            edgecolor="white",
            linewidth=0.3,
            label=CLASS_NAMES[int(class_id)],
            height=0.68,
            zorder=3,
        )
        left += values
    ax.set_yticks(y, ordered_names)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP value|)  (average impact on model output magnitude)")
    legend = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), **LEGEND_KWARGS)
    style_legend_frame(legend)
    style_axes(ax, grid_axis="x")
    _add_plain_panel_tag(ax, "(a)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_interaction_summary(
    interactions: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    selected_feature_names: list[str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    rng = np.random.default_rng(44)
    n_rows = len(selected_feature_names)
    n_cols = len(selected_feature_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.5 * CM, 15.5 * CM), sharex=True, sharey=True, constrained_layout=True)
    axes_array = np.asarray(axes).reshape(n_rows, n_cols)
    y = np.asarray(labels, dtype=np.int64)
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes_array[row, col]
            x_values = interactions[:, row, col]
            jitter = rng.normal(0.0, 0.045, size=x_values.shape[0])
            for class_id in CLASS_ORDER:
                mask = y == int(class_id)
                ax.scatter(
                    x_values[mask],
                    np.full(mask.sum(), n_rows - 1 - row, dtype=np.float32) + jitter[mask],
                    s=12,
                    color=CLASS_COLORS[int(class_id)],
                    alpha=0.95,
                    linewidths=0,
                    zorder=3,
                )
            ax.axvline(0.0, color="#7F7F7F", linewidth=0.8, zorder=2)
            style_axes(ax, grid_axis="y")
            span = float(np.nanmax(np.abs(interactions))) if interactions.size else 0.1
            lim = max(0.15, min(0.65, span * 1.20))
            ax.set_xlim(-lim, lim)
            if row == 0:
                ax.set_title(selected_feature_names[col], fontsize=FONT_SIZE)
            if col == 0:
                ax.set_yticks(range(n_rows), list(reversed(selected_feature_names)))
            else:
                ax.set_yticks(range(n_rows), [])
            if row != n_rows - 1:
                ax.tick_params(labelbottom=False)
    _add_plain_panel_tag(axes_array[0, 0], "(b)")
    fig.supxlabel("SHAP interaction value", y=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_bar_csv(classwise_matrix: np.ndarray[Any, Any], feature_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "class_0", "class_1", "class_2", "total"])
        for idx, name in enumerate(feature_names):
            row = [float(classwise_matrix[idx, class_id]) for class_id in [0, 1, 2]]
            writer.writerow([name, *(f"{value:.4f}" for value in row), f"{sum(row):.4f}"])


def write_meta(
    path: Path,
    *,
    excel_path: Path,
    metrics: dict[str, float],
    bar_path: Path,
    summary_path: Path,
    csv_path: Path,
    raw_feature_names: list[str],
    display_feature_names: list[str],
    selected_feature_names: list[str],
) -> None:
    meta = {
        "analysis": "TreeSHAP EIS feature importance",
        "source_excel": str(excel_path),
        "split": "6:4, seed=44, group holdout-first",
        "model": "RandomForestClassifier trained on 9 normalized EIS statistic features",
        "explainer": "shap.TreeExplainer",
        "metrics": {key: round(float(value), 4) for key, value in metrics.items()},
        "raw_feature_names": raw_feature_names,
        "display_feature_names": display_feature_names,
        "interaction_feature_selection": {
            "rule": "Top-k features by total mean(|SHAP value|) across Class 0, Class 1 and Class 2.",
            "top_k": len(selected_feature_names),
            "selected_features": selected_feature_names,
            "reason": "A full 9 x 9 interaction matrix is difficult to read in a manuscript figure; the 3 x 3 panel follows the provided reference layout and uses the objective SHAP ranking.",
        },
        "bar_figure": str(bar_path),
        "interaction_summary_figure": str(summary_path),
        "csv": str(csv_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", default="data/raw/测试数据.xlsx")
    parser.add_argument("--sheet-name", default="Sheet1")
    parser.add_argument("--output-dir", default="outputs/paper_figures/shape_analysis")
    parser.add_argument("--prefix", default="eis_6_4_shap")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride-train", type=int, default=16)
    parser.add_argument("--stride-val", type=int, default=32)
    parser.add_argument("--stride-eval", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=44)
    parser.add_argument("--test-size", type=float, default=0.40)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--split-mode", default="segment")
    parser.add_argument("--segment-gap-seconds", type=float, default=600.0)
    parser.add_argument("--segment-block-seconds", type=float, default=300.0)
    parser.add_argument("--segment-label-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group-split-strategy", default="holdout_first")
    parser.add_argument("--split-retries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--interaction-top-k", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    excel_path = _as_project_path(args.excel)
    output_dir = _as_project_path(args.output_dir)
    features, labels, split, meta = build_eis_window_features(
        excel_path=excel_path,
        sheet_name=str(args.sheet_name),
        window_size=int(args.window_size),
        stride_train=int(args.stride_train),
        stride_val=int(args.stride_val),
        stride_eval=int(args.stride_eval),
        split_mode=str(args.split_mode),
        segment_gap_seconds=float(args.segment_gap_seconds),
        segment_block_seconds=float(args.segment_block_seconds),
        segment_label_boundary=bool(args.segment_label_boundary),
        random_state=int(args.random_state),
        test_size=float(args.test_size),
        val_size=float(args.val_size),
        group_split_strategy=str(args.group_split_strategy),
        split_retries=int(args.split_retries),
    )
    model, metrics = train_eis_model(features, labels, split, seed=int(args.random_state))
    train_mask = split == 0
    test_mask = split == 2
    impacts = tree_shap_values(model, features[test_mask])
    classwise_matrix = classwise_mean_abs_impact(impacts)
    raw_feature_names = list(meta["feature_names"])
    feature_names = english_feature_names(raw_feature_names)
    selected = top_feature_indices(classwise_matrix, top_k=int(args.interaction_top_k))
    interactions = true_class_interaction_values(model, features[test_mask], labels[test_mask], selected)
    selected_names = [feature_names[int(idx)] for idx in selected]

    bar_path = output_dir / f"{args.prefix}_bar.png"
    summary_path = output_dir / f"{args.prefix}_interaction_summary.png"
    csv_path = output_dir / f"{args.prefix}_bar_values.csv"
    meta_path = output_dir / f"{args.prefix}.meta.json"
    plot_bar(classwise_matrix, feature_names, bar_path, top_k=int(args.top_k))
    plot_interaction_summary(interactions, labels[test_mask], selected_names, summary_path)
    write_bar_csv(classwise_matrix, feature_names, csv_path)
    write_meta(
        meta_path,
        excel_path=excel_path,
        metrics=metrics,
        bar_path=bar_path,
        summary_path=summary_path,
        csv_path=csv_path,
        raw_feature_names=raw_feature_names,
        display_feature_names=feature_names,
        selected_feature_names=selected_names,
    )
    print(
        json.dumps(
            {
                "bar": str(bar_path),
                "interaction_summary": str(summary_path),
                "csv": str(csv_path),
                "meta": str(meta_path),
                "metrics": metrics,
                "top_interaction_features": selected_names,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
