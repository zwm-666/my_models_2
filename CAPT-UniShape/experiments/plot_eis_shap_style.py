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
    COND_COLS,
    EIS_COLS,
    LABEL_COL,
    STACK_COLS,
    TIME_COL,
    _choose_group_split,
    _derive_group_keys,
    _derive_segment_group_keys,
    normalize_columns,
)
from plot_style import (  # noqa: E402
    CM,
    FONT_SIZE,
    FULL_FIG_WIDTH_CM,
    LEGEND_KWARGS,
    MODEL_COLORS,
    PANEL_TAG_POSITION,
    apply_paper_style,
    save_paper_figure,
    style_axes,
    style_legend_frame,
)


CLASS_ORDER = [0, 2, 1]
CLASS_COLORS = {2: MODEL_COLORS[3], 0: MODEL_COLORS[0], 1: MODEL_COLORS[2]}
CLASS_NAMES = {0: "Normal", 1: "Flooding", 2: "Drying"}
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
    "电堆总电压": "Stack voltage",
    "电堆总电流": "Stack current",
    "电堆功率": "Stack power",
    "进堆空压": "Air inlet pressure",
    "出堆水温": "Water outlet temperature",
    "高压水泵转速FK": "Water pump speed",
    "氢压差": "Hydrogen pressure drop",
    "空压差": "Air pressure drop",
    "水温升": "Water temperature rise",
    "FC空压机出口温度": "Compressor outlet temperature",
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


def build_window_mean_features(
    *,
    excel_path: Path,
    sheet_name: str,
    feature_columns: list[str],
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
    normalization_label: str,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]], dict[str, Any]]:
    frame = _read_excel_with_labels(excel_path, sheet_name)
    missing = set(feature_columns + [TIME_COL, LABEL_COL]) - set(frame.columns)
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
    frame[feature_columns] = frame[feature_columns].fillna(0)
    train_frame = frame[frame["__group_key__"].isin(list(split_group_sets[0]))].copy()
    _, means, stds = normalize_columns(train_frame, feature_columns)
    for col in feature_columns:
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
            values = group_frame[feature_columns].to_numpy(dtype=np.float32)
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
        "feature_names": list(feature_columns),
        "split_quality": split_quality,
        "normalization": normalization_label,
    }
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(split, dtype=np.int64),
        meta,
    )


def build_xop_window_features(**kwargs: Any) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]], dict[str, Any]]:
    return build_window_mean_features(
        feature_columns=list(STACK_COLS),
        normalization_label="z-score stack operational variables using 6:4 training groups",
        **kwargs,
    )


def build_xcond_window_features(**kwargs: Any) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]], dict[str, Any]]:
    return build_window_mean_features(
        feature_columns=list(COND_COLS),
        normalization_label="z-score condition variables using 6:4 training groups",
        **kwargs,
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


def feature_order_by_total_impact(classwise_matrix: np.ndarray[Any, Any], top_k: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    return top_feature_indices(classwise_matrix, top_k=top_k)


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
        0.02,
        1.04,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="normal",
    )


def _rank_normalized(values: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    raw_values = np.asarray(values, dtype=np.float32)
    finite_mask = np.isfinite(raw_values)
    if finite_mask.sum() <= 1 or np.isclose(np.nanmin(raw_values), np.nanmax(raw_values)):
        return np.full_like(raw_values, 0.5, dtype=np.float32)
    normalized = np.full_like(raw_values, 0.5, dtype=np.float32)
    valid_values = raw_values[finite_mask]
    sorter = np.argsort(valid_values, kind="mergesort")
    sorted_values = valid_values[sorter]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    group_ends = np.r_[group_starts[1:], sorted_values.size]
    sorted_ranks = np.empty(sorted_values.size, dtype=np.float32)
    for start, end in zip(group_starts, group_ends):
        sorted_ranks[start:end] = (start + end - 1) / 2.0
    ranks = np.empty_like(sorted_ranks)
    ranks[sorter] = sorted_ranks
    normalized[finite_mask] = ranks / float(valid_values.size - 1)
    return normalized.astype(np.float32)


def _draw_stacked_bar_panel(
    ax: Any,
    classwise_matrix: np.ndarray[Any, Any],
    feature_names: list[str],
    *,
    order: np.ndarray[Any, np.dtype[np.int64]],
) -> None:
    ordered_names = [feature_names[int(idx)] for idx in order]
    ordered_matrix = np.asarray(classwise_matrix, dtype=np.float32)[order]
    y = np.arange(len(order))
    left = np.zeros(len(order), dtype=np.float32)
    bar_height = 0.42 if len(order) <= 4 else 0.54
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
            height=bar_height,
            zorder=3,
        )
        left += values
    ax.set_yticks(y, ordered_names)
    ax.invert_yaxis()
    if len(order) <= 4:
        ax.set_ylim(len(order) + 0.58, -0.50)
    ax.set_xlabel("mean(|SHAP value|)")
    style_axes(ax, grid_axis="x")
    ax.tick_params(axis="both", labelsize=6.4)
    ax.xaxis.label.set_size(7.5)
    _add_plain_panel_tag(ax, "(a)")


def _draw_beeswarm_panel(
    ax: Any,
    shap_values: np.ndarray[Any, Any],
    feature_values: np.ndarray[Any, Any],
    feature_names: list[str],
    *,
    order: np.ndarray[Any, np.dtype[np.int64]],
    rng_seed: int,
) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    rng = np.random.default_rng(int(rng_seed))
    values = np.asarray(shap_values, dtype=np.float32)
    features = np.asarray(feature_values, dtype=np.float32)
    ordered_names = [feature_names[int(idx)] for idx in order]
    cmap = LinearSegmentedColormap.from_list("feature_value", ["#2F5DA8", "#F0F0F0", "#D73027"])
    for row_pos, feature_idx in enumerate(order[::-1]):
        x_values = values[:, int(feature_idx)]
        normalized = _rank_normalized(features[:, int(feature_idx)])
        jitter = rng.normal(0.0, 0.075, size=x_values.shape[0])
        ax.scatter(
            x_values,
            np.full(x_values.shape[0], row_pos, dtype=np.float32) + jitter,
            c=normalized,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=6,
            alpha=0.95,
            linewidths=0,
            zorder=3,
        )
    ax.axvline(0.0, color="#7F7F7F", linewidth=0.8, zorder=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([])
    ax.set_xlabel("SHAP value")
    style_axes(ax, grid_axis="x")
    ax.tick_params(axis="both", labelsize=6.4)
    ax.xaxis.label.set_size(7.5)
    _add_plain_panel_tag(ax, "(b)")
    return plt.cm.ScalarMappable(cmap=cmap)


def plot_combined_shap_figure(
    classwise_matrix: np.ndarray[Any, Any],
    true_class_impacts: np.ndarray[Any, Any],
    feature_values: np.ndarray[Any, Any],
    feature_names: list[str],
    output_base: Path,
    *,
    top_k: int,
    rng_seed: int = 44,
) -> list[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    order = feature_order_by_total_impact(classwise_matrix, top_k=top_k)
    selected_names = [feature_names[int(idx)] for idx in order]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_FIG_WIDTH_CM * CM, 7.9 * CM),
        gridspec_kw={"width_ratios": [1.06, 1.0], "wspace": 0.30},
    )
    _draw_stacked_bar_panel(axes[0], classwise_matrix, feature_names, order=order)
    mappable = _draw_beeswarm_panel(axes[1], true_class_impacts, feature_values, feature_names, order=order, rng_seed=rng_seed)
    legend_kwargs = dict(LEGEND_KWARGS)
    legend_kwargs.update(fontsize=6.4, borderpad=0.25, labelspacing=0.18, handletextpad=0.25)
    legend = axes[0].legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), **legend_kwargs)
    style_legend_frame(legend)
    cbar = fig.colorbar(mappable, ax=axes[1], fraction=0.045, pad=0.03)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value")
    cbar.ax.tick_params(labelsize=6.4)
    cbar.ax.yaxis.label.set_size(7.5)
    fig.subplots_adjust(left=0.15, right=0.94, bottom=0.16, top=0.88)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, str(output_base), formats=("png", "pdf", "svg"))
    plt.close(fig)
    return selected_names


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


def true_class_shap_values(
    shap_values: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    classes: np.ndarray[Any, Any],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return per-sample SHAP values for each sample's true class."""
    values = np.asarray(shap_values, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    class_positions = {int(cls): idx for idx, cls in enumerate(np.asarray(classes, dtype=np.int64).tolist())}
    out = np.zeros((values.shape[0], values.shape[1]), dtype=np.float32)
    for sample_idx, label in enumerate(y.tolist()):
        out[sample_idx] = values[sample_idx, :, class_positions[int(label)]]
    return out


def plot_feature_shap_summary(
    shap_values: np.ndarray[Any, Any],
    feature_values: np.ndarray[Any, Any],
    feature_names: list[str],
    output_path: Path,
    *,
    top_k: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    _setup_matplotlib()
    rng = np.random.default_rng(44)
    values = np.asarray(shap_values, dtype=np.float32)
    features = np.asarray(feature_values, dtype=np.float32)
    order = np.argsort(-np.mean(np.abs(values), axis=0))[: int(top_k)]
    ordered_names = [feature_names[int(idx)] for idx in order]
    n_rows = len(order)
    fig, ax = plt.subplots(figsize=(18 * CM, 11 * CM), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("feature_value", ["#2F5DA8", "#F0F0F0", "#D73027"])
    for row_pos, feature_idx in enumerate(order[::-1]):
        x_values = values[:, int(feature_idx)]
        raw_values = features[:, int(feature_idx)]
        finite_mask = np.isfinite(raw_values)
        if finite_mask.sum() <= 1 or np.isclose(np.nanmin(raw_values), np.nanmax(raw_values)):
            normalized = np.full_like(raw_values, 0.5, dtype=np.float32)
        else:
            normalized = np.full_like(raw_values, 0.5, dtype=np.float32)
            valid_values = raw_values[finite_mask]
            sorter = np.argsort(valid_values, kind="mergesort")
            sorted_values = valid_values[sorter]
            group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
            group_ends = np.r_[group_starts[1:], sorted_values.size]
            sorted_ranks = np.empty(sorted_values.size, dtype=np.float32)
            for start, end in zip(group_starts, group_ends):
                sorted_ranks[start:end] = (start + end - 1) / 2.0
            ranks = np.empty_like(sorted_ranks)
            ranks[sorter] = sorted_ranks
            normalized[finite_mask] = ranks / float(valid_values.size - 1)
        jitter = rng.normal(0.0, 0.075, size=x_values.shape[0])
        ax.scatter(
            x_values,
            np.full(x_values.shape[0], row_pos, dtype=np.float32) + jitter,
            c=normalized,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=12,
            alpha=0.95,
            linewidths=0,
            zorder=3,
        )
    ax.axvline(0.0, color="#7F7F7F", linewidth=0.8, zorder=2)
    ax.set_yticks(range(n_rows), list(reversed(ordered_names)))
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Original EIS feature")
    style_axes(ax, grid_axis="x")
    _add_plain_panel_tag(ax, "(b)")
    cbar_ax = ax.inset_axes([1.018, 0.12, 0.026, 0.76], transform=ax.transAxes)
    n_steps = 96
    for step in range(n_steps):
        y0 = step / n_steps
        cbar_ax.add_patch(
            Rectangle(
                (0.0, y0),
                1.0,
                1.0 / n_steps,
                facecolor=cmap((step + 0.5) / n_steps),
                edgecolor="none",
            )
        )
    cbar_ax.set_xlim(0.0, 1.0)
    cbar_ax.set_ylim(0.0, 1.0)
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([0.0, 1.0])
    cbar_ax.set_yticklabels(["Low", "High"])
    cbar_ax.yaxis.tick_right()
    cbar_ax.yaxis.set_label_position("right")
    cbar_ax.set_ylabel("Feature value", rotation=90, labelpad=6, fontsize=FONT_SIZE)
    cbar_ax.tick_params(axis="y", labelsize=FONT_SIZE, length=2.5, width=0.7)
    for spine in cbar_ax.spines.values():
        spine.set_linewidth(0.7)
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
    analysis_name: str,
    model_name: str,
    metrics: dict[str, float],
    figure_path: Path,
    csv_path: Path,
    raw_feature_names: list[str],
    display_feature_names: list[str],
    summary_feature_names: list[str],
) -> None:
    meta = {
        "analysis": analysis_name,
        "source_excel": str(excel_path),
        "split": "6:4, seed=44, group holdout-first",
        "model": model_name,
        "explainer": "shap.TreeExplainer",
        "metrics": {key: round(float(value), 4) for key, value in metrics.items()},
        "raw_feature_names": raw_feature_names,
        "display_feature_names": display_feature_names,
        "feature_shap_summary": {
            "rule": "Top-k features by total classwise mean absolute SHAP value.",
            "top_k": len(summary_feature_names),
            "selected_features": summary_feature_names,
            "color": "Per-feature rank-normalized raw feature value on the test set.",
        },
        "combined_figure": str(figure_path),
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
    parser.add_argument("--modalities", default="all", choices=["all", "both", "eis", "x_op", "x_cond", "condition"], help="Which SHAP figures to generate.")
    return parser


def _common_build_kwargs(args: argparse.Namespace, excel_path: Path) -> dict[str, Any]:
    return {
        "excel_path": excel_path,
        "sheet_name": str(args.sheet_name),
        "window_size": int(args.window_size),
        "stride_train": int(args.stride_train),
        "stride_val": int(args.stride_val),
        "stride_eval": int(args.stride_eval),
        "split_mode": str(args.split_mode),
        "segment_gap_seconds": float(args.segment_gap_seconds),
        "segment_block_seconds": float(args.segment_block_seconds),
        "segment_label_boundary": bool(args.segment_label_boundary),
        "random_state": int(args.random_state),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "group_split_strategy": str(args.group_split_strategy),
        "split_retries": int(args.split_retries),
    }


def run_shap_analysis(
    *,
    modality: str,
    features: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    split: np.ndarray[Any, Any],
    raw_feature_names: list[str],
    output_dir: Path,
    prefix: str,
    excel_path: Path,
    top_k: int,
    random_state: int,
    analysis_name: str,
    model_name: str,
) -> dict[str, Any]:
    model, metrics = train_eis_model(features, labels, split, seed=int(random_state))
    test_mask = split == 2
    impacts = tree_shap_values(model, features[test_mask])
    classwise_matrix = classwise_mean_abs_impact(impacts)
    feature_names = english_feature_names(raw_feature_names)
    true_class_impacts = true_class_shap_values(impacts, labels[test_mask], model.classes_)
    output_base = output_dir / f"{prefix}_{modality}_combined"
    csv_path = output_dir / f"{prefix}_{modality}_bar_values.csv"
    meta_path = output_dir / f"{prefix}_{modality}.meta.json"
    selected_names = plot_combined_shap_figure(
        classwise_matrix,
        true_class_impacts,
        features[test_mask],
        feature_names,
        output_base,
        top_k=min(int(top_k), len(feature_names)),
        rng_seed=int(random_state),
    )
    write_bar_csv(classwise_matrix, feature_names, csv_path)
    write_meta(
        meta_path,
        excel_path=excel_path,
        analysis_name=analysis_name,
        model_name=model_name,
        metrics=metrics,
        figure_path=output_base.with_suffix(".png"),
        csv_path=csv_path,
        raw_feature_names=raw_feature_names,
        display_feature_names=feature_names,
        summary_feature_names=selected_names,
    )
    return {
        "modality": modality,
        "figure": str(output_base.with_suffix(".png")),
        "svg": str(output_base.with_suffix(".svg")),
        "pdf": str(output_base.with_suffix(".pdf")),
        "csv": str(csv_path),
        "meta": str(meta_path),
        "metrics": metrics,
        "top_features": selected_names,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    excel_path = _as_project_path(args.excel)
    output_dir = _as_project_path(args.output_dir)
    outputs: list[dict[str, Any]] = []
    common_kwargs = _common_build_kwargs(args, excel_path)
    modalities = "x_cond" if str(args.modalities) == "condition" else str(args.modalities)
    if modalities in {"all", "both", "eis"}:
        features, labels, split, meta = build_eis_window_features(**common_kwargs)
        outputs.append(
            run_shap_analysis(
                modality="eis",
                features=features,
                labels=labels,
                split=split,
                raw_feature_names=list(meta["feature_names"]),
                output_dir=output_dir,
                prefix=str(args.prefix),
                excel_path=excel_path,
                top_k=int(args.top_k),
                random_state=int(args.random_state),
                analysis_name="TreeSHAP EIS feature importance",
                model_name="RandomForestClassifier trained on 9 normalized EIS statistic features",
            )
        )
    if modalities in {"all", "both", "x_op"}:
        features, labels, split, meta = build_xop_window_features(**common_kwargs)
        outputs.append(
            run_shap_analysis(
                modality="x_op",
                features=features,
                labels=labels,
                split=split,
                raw_feature_names=list(meta["feature_names"]),
                output_dir=output_dir,
                prefix=str(args.prefix),
                excel_path=excel_path,
                top_k=min(int(args.top_k), len(meta["feature_names"])),
                random_state=int(args.random_state),
                analysis_name="TreeSHAP x_op operational feature importance",
                model_name="RandomForestClassifier trained on normalized x_op stack operational window-mean features",
            )
        )
    if modalities in {"all", "x_cond"}:
        features, labels, split, meta = build_xcond_window_features(**common_kwargs)
        outputs.append(
            run_shap_analysis(
                modality="x_cond",
                features=features,
                labels=labels,
                split=split,
                raw_feature_names=list(meta["feature_names"]),
                output_dir=output_dir,
                prefix=str(args.prefix),
                excel_path=excel_path,
                top_k=min(int(args.top_k), len(meta["feature_names"])),
                random_state=int(args.random_state),
                analysis_name="TreeSHAP condition feature importance",
                model_name="RandomForestClassifier trained on normalized x_cond condition window-mean features",
            )
        )
    print(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
