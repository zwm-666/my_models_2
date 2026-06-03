"""Visualize condition-variable screening for CAPT-UniShape."""

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
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "paper_figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FIG_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_DIR))

from plot_style import (  # noqa: E402
    ANNOTATION_SIZE,
    CM,
    FULL_FIG_WIDTH_CM,
    LEGEND_KWARGS,
    MODEL_COLORS,
    SUBPLOT_MARGINS_TOP_LEGEND,
    TOP_LEGEND_ANCHOR_Y,
    apply_paper_style,
    save_paper_figure,
    style_axes,
    style_legend_frame,
)
from experiments.build_official_npz_from_self_excel import _choose_group_split, _derive_group_keys, _derive_segment_group_keys  # noqa: E402
from src.datasets.self_dataset import COND_COLS, EIS_COLS, STACK_COLS, TIME_COL  # noqa: E402


RAW_EXCEL = ROOT / "data" / "raw" / "水淹和膜干故障测试数据_补充特征汇总.xlsx"
OUT_DIR = FIG_DIR / "condition_selection"
LABEL_ALIASES = ["label", "Label", "标签", "类型"]
RAW_LABEL_MAP = {"正常": 0, "过湿": 1, "水淹": 1, "过干": 2, "膜干": 2}

COND_DISPLAY = {
    "FC系统入口高压": "System inlet pressure",
    "进堆氢压": "H2 inlet pressure",
    "出堆氢压": "H2 outlet pressure",
    "进堆空压": "Air inlet pressure",
    "FC空压机出口压力": "Compressor outlet pressure",
    "进堆空温": "Air inlet temperature",
    "出堆空温": "Air outlet temperature",
    "进堆水温": "Water inlet temperature",
    "出堆水温": "Water outlet temperature",
    "氢气循环泵FK": "Hydrogen pump feedback",
    "比例阀反馈": "Proportional valve feedback",
    "purge时间": "Purge duration",
    "离心机速度FK": "Centrifuge speed feedback",
    "离心机功率": "Centrifuge power",
    "三通阀开度FK": "Three-way valve opening",
    "高压水泵转速FK": "Water pump speed",
    "FC空压机出口温度": "Compressor outlet temperature",
    "FC换热器入口温度": "Heat-exchanger inlet temperature",
    "氢压差": "Hydrogen pressure drop",
    "空压差": "Air pressure drop",
    "空温升": "Air temperature rise",
    "水温升": "Water temperature rise",
}


def _label_column(frame: pd.DataFrame) -> str:
    for col in LABEL_ALIASES:
        if col in frame.columns:
            return str(col)
    raise ValueError(f"Missing label column. Expected one of {LABEL_ALIASES}")


def english_condition_names(columns: list[str]) -> list[str]:
    return [COND_DISPLAY.get(str(col), str(col)) for col in columns]


def condition_candidate_columns(frame: pd.DataFrame) -> list[str]:
    label_col = _label_column(frame)
    excluded = {TIME_COL, label_col, "label", "Label", "标签", "类型", "__label__", *EIS_COLS, *STACK_COLS}
    return [str(col) for col in frame.select_dtypes(include=[np.number]).columns if str(col) not in excluded]


def _training_group_mask(
    frame: pd.DataFrame,
    *,
    split_mode: str,
    segment_gap_seconds: float,
    segment_block_seconds: float,
    segment_label_boundary: bool,
    random_state: int,
    test_size: float,
    val_size: float,
    group_split_strategy: str,
    split_retries: int,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    working = frame.copy()
    if split_mode == "segment":
        working["__group_key__"] = _derive_segment_group_keys(
            working,
            gap_seconds=float(segment_gap_seconds),
            block_seconds=float(segment_block_seconds),
            label_boundary=bool(segment_label_boundary),
        )
    else:
        working["__group_key__"] = _derive_group_keys(pd.Series(working[TIME_COL].to_numpy()), split_mode)
    groups = np.asarray(working["__group_key__"].unique())
    group_label_map = working.groupby("__group_key__")["__label__"].first()
    group_labels = np.array([group_label_map[group] for group in groups])
    row_counts = {group: int(count) for group, count in working.groupby("__group_key__").size().items()}
    train_groups, _val_groups, _test_groups, _quality = _choose_group_split(
        groups=groups,
        group_labels=group_labels,
        row_counts=row_counts,
        group_label_map=group_label_map,
        test_size=float(test_size),
        val_size=float(val_size),
        random_state=int(random_state),
        group_split_strategy=group_split_strategy,
        window_size=64,
        stride_train=16,
        stride_val=32,
        stride_eval=32,
        split_retries=int(split_retries),
        min_eval_class_windows=5,
        min_eval_class_groups=1,
        min_train_class_windows=1,
        min_train_class_groups=1,
        min_val_class_groups=1,
        min_test_class_groups=1,
        prefer_balanced_train_groups=False,
    )
    return working["__group_key__"].isin(set(train_groups.tolist())).to_numpy(dtype=bool)


def prepare_condition_frame(
    excel_path: Path,
    sheet_name: str,
    *,
    split_mode: str = "segment",
    segment_gap_seconds: float = 600.0,
    segment_block_seconds: float = 300.0,
    segment_label_boundary: bool = True,
    random_state: int = 44,
    test_size: float = 0.40,
    val_size: float = 0.25,
    group_split_strategy: str = "holdout_first",
    split_retries: int = 50,
) -> tuple[pd.DataFrame, np.ndarray[Any, np.dtype[np.int64]], list[str], dict[str, Any]]:
    frame = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    label_col = _label_column(frame)
    labels_raw = frame[label_col]
    if labels_raw.dtype == object:
        labels = labels_raw.astype(str).str.strip().map(RAW_LABEL_MAP)
        if labels.isna().any():
            unknown = sorted(set(labels_raw.astype(str).str.strip()) - set(RAW_LABEL_MAP))
            raise ValueError(f"Unknown labels: {unknown}")
        y = labels.to_numpy(dtype=np.int64)
    else:
        y = labels_raw.to_numpy(dtype=np.int64)
    frame = frame.copy()
    frame["__label__"] = y
    candidates = condition_candidate_columns(frame)
    if not candidates:
        raise ValueError("No numeric condition candidate columns found.")
    train_mask = _training_group_mask(
        frame,
        split_mode=split_mode,
        segment_gap_seconds=segment_gap_seconds,
        segment_block_seconds=segment_block_seconds,
        segment_label_boundary=segment_label_boundary,
        random_state=random_state,
        test_size=test_size,
        val_size=val_size,
        group_split_strategy=group_split_strategy,
        split_retries=split_retries,
    )
    values = frame.loc[train_mask, candidates].fillna(0).astype(float)
    meta = {
        "screening_rows": int(train_mask.sum()),
        "screening_scope": "training groups only",
        "split": f"{int(round((1.0 - test_size) * 10))}:{int(round(test_size * 10))}, seed={random_state}",
    }
    return values, y[train_mask], candidates, meta


def condition_screening_scores(
    values: pd.DataFrame,
    labels: np.ndarray[Any, np.dtype[np.int64]],
    *,
    seed: int,
) -> pd.DataFrame:
    scaler = StandardScaler()
    x = scaler.fit_transform(values.to_numpy(dtype=np.float32))
    y = np.asarray(labels, dtype=np.int64)
    forest = RandomForestClassifier(
        n_estimators=500,
        random_state=int(seed),
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    forest.fit(x, y)
    f_values, p_values = f_classif(x, y)
    score = pd.DataFrame(
        {
            "feature": list(values.columns),
            "display_name": english_condition_names(list(values.columns)),
            "selected": [str(col) in set(COND_COLS) for col in values.columns],
            "rf_importance": forest.feature_importances_.astype(float),
            "anova_f": np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0).astype(float),
            "anova_p": np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0).astype(float),
        }
    )
    score["rf_rank"] = score["rf_importance"].rank(ascending=False, method="min").astype(int)
    score["anova_rank"] = score["anova_f"].rank(ascending=False, method="min").astype(int)
    n_features = max(len(score), 1)
    score["rf_rank_score"] = (n_features - score["rf_rank"] + 1) / n_features
    score["anova_rank_score"] = (n_features - score["anova_rank"] + 1) / n_features
    score["fused_relevance"] = 0.5 * score["rf_rank_score"] + 0.5 * score["anova_rank_score"]
    return score.sort_values(["selected", "rf_importance"], ascending=[False, False]).reset_index(drop=True)


def add_redundancy_to_selected(scores: pd.DataFrame, values: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    corr = values.corr().abs()
    selected_set = {feature for feature in selected_features if feature in corr.columns}
    rows = []
    for row in scores.to_dict("records"):
        feature = str(row["feature"])
        if feature in selected_set:
            references = [item for item in selected_set if item != feature]
        else:
            references = list(selected_set)
        if not references:
            row["max_corr_to_selected"] = 0.0
            row["nearest_selected"] = ""
            row["nearest_selected_display"] = ""
        else:
            nearest = max(references, key=lambda item: float(corr.loc[feature, item]))
            row["max_corr_to_selected"] = float(corr.loc[feature, nearest])
            row["nearest_selected"] = nearest
            row["nearest_selected_display"] = english_condition_names([nearest])[0]
        rows.append(row)
    return pd.DataFrame(rows)


def clustered_correlation_order(values: pd.DataFrame, columns: list[str]) -> list[str]:
    corr = values[columns].corr().abs().fillna(0.0)
    if len(columns) <= 2:
        return columns
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        distance = 1.0 - corr.to_numpy(dtype=float)
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        order = leaves_list(linkage(condensed, method="average"))
        return [columns[int(idx)] for idx in order]
    except Exception:
        return columns


def _wrap_labels(labels: list[str], width: int = 22) -> list[str]:
    import textwrap

    return ["\n".join(textwrap.wrap(label, width=width, break_long_words=False)) for label in labels]


def _set_tick_label_size(ax, size: float) -> None:
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontsize(size)


def _sparse_bar_codes(plot_frame: pd.DataFrame, value_column: str, *, top_k: int = 5) -> list[str]:
    important = set(plot_frame.loc[plot_frame["selected"], "feature"].astype(str))
    important.update(plot_frame.sort_values(value_column, ascending=False).head(top_k)["feature"].astype(str))
    return [str(row["code"]) if str(row["feature"]) in important else "" for _, row in plot_frame.iterrows()]


def _annotate_bar_codes(ax, plot_frame: pd.DataFrame, value_column: str, *, top_k: int = 5) -> None:
    labels = _sparse_bar_codes(plot_frame, value_column, top_k=top_k)
    x_values = plot_frame[value_column].to_numpy(dtype=float)
    x_limit = max(float(np.nanmax(x_values)) * 1.10, 1.0)
    ax.set_xlim(0.0, x_limit)
    for y, (label, value) in enumerate(zip(labels, x_values)):
        if not label:
            continue
        ax.text(
            min(float(value) + x_limit * 0.012, x_limit * 0.985),
            y,
            label,
            ha="left" if float(value) < x_limit * 0.94 else "right",
            va="center",
            fontsize=6.0,
            clip_on=True,
        )


def plot_condition_selection(scores: pd.DataFrame, values: pd.DataFrame, output_base: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    apply_paper_style()
    plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "path"})

    scores = add_redundancy_to_selected(scores, values, COND_COLS)
    selected_color = MODEL_COLORS[0]
    candidate_color = MODEL_COLORS[5]
    redundancy_color = MODEL_COLORS[2]
    code_order = scores.sort_values("feature")["feature"].tolist()
    feature_codes = {feature: f"C{idx + 1}" for idx, feature in enumerate(code_order)}
    code_names = {
        feature_codes[row["feature"]]: str(row["display_name"])
        for row in scores[["feature", "display_name"]].to_dict("records")
    }

    top_scores = scores.sort_values("rf_importance", ascending=True).copy()
    top_scores["code"] = top_scores["feature"].map(feature_codes)
    y_pos = np.arange(len(top_scores))
    bar_colors = [selected_color if selected else candidate_color for selected in top_scores["selected"]]

    selected_columns = [col for col in COND_COLS if col in values.columns]
    remaining_columns = [col for col in values.columns if col not in selected_columns]
    corr_columns = clustered_correlation_order(values, selected_columns + remaining_columns)
    corr_matrix = values[corr_columns].corr().abs().to_numpy(dtype=float)
    corr_labels = [feature_codes[col] for col in corr_columns]

    fig = plt.figure(figsize=(FULL_FIG_WIDTH_CM * CM, 25.8 * CM))
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[2.55, 0.92, 2.05, 0.90],
        hspace=0.42,
        wspace=0.32,
    )
    ax_rf = fig.add_subplot(grid[0, 0])
    ax_anova = fig.add_subplot(grid[0, 1])
    ax_redundancy = fig.add_subplot(grid[1, :])
    ax_corr = fig.add_subplot(grid[2, :])
    ax_key = fig.add_subplot(grid[3, :])

    ax_rf.barh(y_pos, top_scores["rf_importance"], color=bar_colors, edgecolor="white", linewidth=0.3, zorder=3)
    ax_rf.set_yticks(y_pos, top_scores["code"].tolist())
    ax_rf.set_xlabel("Random forest importance", labelpad=2.0)
    ax_rf.set_ylabel("Candidate variables")
    style_axes(ax_rf, grid_axis="x")
    ax_rf.tick_params(axis="y", pad=2)
    _set_tick_label_size(ax_rf, 5.8)
    ax_rf.text(0.02, 1.005, "(a)", transform=ax_rf.transAxes, ha="left", va="bottom", fontweight="normal")

    anova_plot = scores.sort_values("anova_f", ascending=True).copy()
    anova_plot["code"] = anova_plot["feature"].map(feature_codes)
    y_anova = np.arange(len(anova_plot))
    anova_colors = [selected_color if selected else candidate_color for selected in anova_plot["selected"]]
    ax_anova.barh(y_anova, anova_plot["anova_f"], color=anova_colors, edgecolor="white", linewidth=0.3, zorder=3)
    ax_anova.set_yticks(y_anova, anova_plot["code"].tolist())
    ax_anova.set_xlabel("ANOVA F-score", labelpad=2.0)
    ax_anova.set_ylabel("Candidate variables")
    style_axes(ax_anova, grid_axis="x")
    ax_anova.tick_params(axis="y", pad=2)
    _set_tick_label_size(ax_anova, 5.8)
    ax_anova.text(0.02, 1.005, "(b)", transform=ax_anova.transAxes, ha="left", va="bottom", fontweight="normal")

    image = ax_corr.imshow(corr_matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax_corr.set_xticks(np.arange(len(corr_columns)), corr_labels, rotation=90, ha="center")
    ax_corr.set_yticks(np.arange(len(corr_columns)), corr_labels)
    selected_code_positions = [idx for idx, col in enumerate(corr_columns) if col in set(COND_COLS)]
    decision_scores = scores.copy()
    decision_scores["code"] = decision_scores["feature"].map(feature_codes)
    redundancy_plot = (
        decision_scores[~decision_scores["selected"]]
        .sort_values("fused_relevance", ascending=False)
        .head(10)
        .sort_values("max_corr_to_selected", ascending=True)
        .copy()
    )
    redundancy_plot["nearest_code"] = redundancy_plot["nearest_selected"].map(feature_codes)
    y_red = np.arange(len(redundancy_plot))
    ax_redundancy.barh(y_red, redundancy_plot["max_corr_to_selected"], color=candidate_color, edgecolor="white", linewidth=0.3, zorder=3)
    ax_redundancy.set_yticks(y_red, [f"{row.code} -> {row.nearest_code}" for row in redundancy_plot.itertuples()])
    ax_redundancy.axvline(0.90, color=redundancy_color, linestyle="--", linewidth=0.9)
    for y, row in zip(y_red, redundancy_plot.itertuples()):
        value = float(row.max_corr_to_selected)
        ax_redundancy.text(
            value + 0.010,
            y,
            f"{value:.3f}",
            fontsize=7.0,
            ha="left",
            va="center",
            clip_on=False,
        )
    ax_redundancy.text(0.90, 1.02, "|r| = 0.90", transform=ax_redundancy.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.2)
    ax_redundancy.set_xlim(0.0, 1.10)
    ax_redundancy.set_xlabel("Max |r| to current x_cond variable")
    ax_redundancy.set_ylabel("Candidate -> nearest x_cond")
    style_axes(ax_redundancy, grid_axis="x")
    _set_tick_label_size(ax_redundancy, 7.2)
    ax_redundancy.text(0.02, 1.08, "(c)", transform=ax_redundancy.transAxes, ha="left", va="bottom", fontweight="normal")

    ax_corr.set_title("Clustered correlation map", pad=10)
    for idx, col in enumerate(corr_columns):
        if col in set(COND_COLS):
            ax_corr.add_patch(plt.Rectangle((idx - 0.5, idx - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.2))
    for idx in selected_code_positions:
        ax_corr.axvline(idx, color="white", linewidth=0.3, alpha=0.55)
        ax_corr.axhline(idx, color="white", linewidth=0.3, alpha=0.55)
    style_axes(ax_corr, grid_axis="none")
    ax_corr.tick_params(axis="x", pad=2)
    ax_corr.tick_params(axis="y", pad=2)
    _set_tick_label_size(ax_corr, 6.2)
    ax_corr.text(0.02, 1.08, "(d)", transform=ax_corr.transAxes, ha="left", va="bottom", fontweight="normal")
    cbar = fig.colorbar(image, ax=ax_corr, fraction=0.020, pad=0.02)
    cbar.set_label("|r|")
    ax_corr.set_xlabel("Clustered candidate variables")
    ax_corr.set_ylabel("Clustered candidate variables")

    ax_key.axis("off")
    selected_set = set(COND_COLS)
    key_rows = []
    for feature in code_order:
        code = feature_codes[feature]
        marker = "*" if feature in selected_set else ""
        key_rows.append(f"{code}{marker}: {code_names[code]}")
    columns = 2
    rows_per_col = int(np.ceil(len(key_rows) / columns))
    for col_idx in range(columns):
        x = 0.02 + col_idx * 0.49
        for row_idx, text in enumerate(key_rows[col_idx * rows_per_col : (col_idx + 1) * rows_per_col]):
            y = 0.98 - row_idx * (0.92 / max(rows_per_col - 1, 1))
            ax_key.text(x, y, text, transform=ax_key.transAxes, ha="left", va="top", fontsize=5.8)
    ax_key.text(
        0.02,
        -0.08,
        "* current x_cond variable. Scores use training groups only; panel (c) shows redundancy to the nearest current x_cond variable.",
        transform=ax_key.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
    )

    handles = [
        Patch(facecolor=selected_color, edgecolor="none", label="Current x_cond variable"),
        Patch(facecolor=candidate_color, edgecolor="none", label="Other candidate"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        **LEGEND_KWARGS,
    )
    style_legend_frame(legend)
    margins = dict(SUBPLOT_MARGINS_TOP_LEGEND)
    margins.update(left=0.12, right=0.98, top=0.90, bottom=0.04)
    fig.subplots_adjust(**margins)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, str(output_base), formats=("png", "pdf", "svg"))
    plt.close(fig)


def write_scores_csv(scores: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "feature",
        "display_name",
        "selected",
        "rf_importance",
        "anova_f",
        "anova_p",
        "rf_rank",
        "anova_rank",
        "fused_relevance",
        "max_corr_to_selected",
        "nearest_selected_display",
    ]
    existing_fields = [field for field in fields if field in scores.columns]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=existing_fields)
        writer.writeheader()
        for row in scores[existing_fields].to_dict("records"):
            formatted = dict(row)
            for key in ["rf_importance", "anova_f", "anova_p", "fused_relevance", "max_corr_to_selected"]:
                if key in formatted:
                    formatted[key] = f"{float(formatted[key]):.4f}"
            writer.writerow(formatted)


def write_meta(path: Path, *, excel_path: Path, figure_base: Path, scores_csv: Path, candidates: list[str], screening_meta: dict[str, Any]) -> None:
    meta = {
        "analysis": "Condition-variable screening visualization",
        "source_excel": str(excel_path),
        "candidate_count": len(candidates),
        "screening_scope": screening_meta.get("screening_scope"),
        "screening_rows": screening_meta.get("screening_rows"),
        "split": screening_meta.get("split"),
        "candidate_variables": english_condition_names(candidates),
        "selected_count": len(COND_COLS),
        "selected_variables": english_condition_names(COND_COLS),
        "screening_rule": "RandomForest importance + ANOVA F-score are converted to rank scores and averaged as fused relevance; candidate variables are then checked against correlation redundancy, with |r| = 0.9 shown as the redundancy reference level.",
        "figure_png": str(figure_base.with_suffix(".png")),
        "figure_pdf": str(figure_base.with_suffix(".pdf")),
        "figure_svg": str(figure_base.with_suffix(".svg")),
        "scores_csv": str(scores_csv),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", default=str(RAW_EXCEL))
    parser.add_argument("--sheet-name", default="Sheet1")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--prefix", default="condition_selection_process")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--test-size", type=float, default=0.40)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--split-mode", default="segment")
    parser.add_argument("--segment-gap-seconds", type=float, default=600.0)
    parser.add_argument("--segment-block-seconds", type=float, default=300.0)
    parser.add_argument("--segment-label-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group-split-strategy", default="holdout_first")
    parser.add_argument("--split-retries", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    excel_path = Path(args.excel)
    if not excel_path.is_absolute():
        excel_path = ROOT / excel_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    values, labels, candidates, screening_meta = prepare_condition_frame(
        excel_path,
        str(args.sheet_name),
        split_mode=str(args.split_mode),
        segment_gap_seconds=float(args.segment_gap_seconds),
        segment_block_seconds=float(args.segment_block_seconds),
        segment_label_boundary=bool(args.segment_label_boundary),
        random_state=int(args.seed),
        test_size=float(args.test_size),
        val_size=float(args.val_size),
        group_split_strategy=str(args.group_split_strategy),
        split_retries=int(args.split_retries),
    )
    scores = condition_screening_scores(values, labels, seed=int(args.seed))
    scores = add_redundancy_to_selected(scores, values, COND_COLS)
    figure_base = output_dir / str(args.prefix)
    scores_csv = output_dir / f"{args.prefix}_scores.csv"
    meta_json = output_dir / f"{args.prefix}.meta.json"
    plot_condition_selection(scores, values, figure_base)
    write_scores_csv(scores, scores_csv)
    write_meta(meta_json, excel_path=excel_path, figure_base=figure_base, scores_csv=scores_csv, candidates=candidates, screening_meta=screening_meta)
    print(json.dumps({"figure": str(figure_base.with_suffix(".png")), "scores_csv": str(scores_csv), "meta": str(meta_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
