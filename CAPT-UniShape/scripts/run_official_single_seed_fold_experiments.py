"""Run single-seed, multi-test-fold official experiments.

This script uses one random seed to create stratified group folds. Each fold has
an independent held-out test group, and every model is retrained per fold.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_converter = importlib.import_module("scripts.build_official_npz_from_self_excel")
_baseline = importlib.import_module("scripts.run_official_baseline_experiments")


def _label_count_dict(values: np.ndarray[Any, Any]) -> dict[str, int]:
    if values.size == 0:
        return {}
    labels, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _make_stratified_group_folds(
    groups: np.ndarray[Any, Any],
    group_labels: np.ndarray[Any, Any],
    n_folds: int,
    seed: int,
    val_fraction: float,
) -> list[dict[str, np.ndarray[Any, Any]]]:
    """Create disjoint stratified test folds using one seed."""
    if int(n_folds) < 2:
        raise ValueError("n_folds must be >= 2")
    labels = np.asarray(group_labels, dtype=np.int64)
    min_class_groups = int(np.min(np.unique(labels, return_counts=True)[1]))
    if int(n_folds) > min_class_groups:
        raise ValueError(f"n_folds={n_folds} exceeds minimum class group count={min_class_groups}")

    splitter = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    folds: list[dict[str, np.ndarray[Any, Any]]] = []
    for train_val_idx, test_idx in splitter.split(groups, labels):
        train_val_groups = np.asarray(groups[train_val_idx])
        train_val_labels = labels[train_val_idx]
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=float(val_fraction),
            stratify=train_val_labels,
            random_state=int(seed),
        )
        folds.append(
            {
                "train": np.asarray(train_groups),
                "val": np.asarray(val_groups),
                "test": np.asarray(groups[test_idx]),
            }
        )
    return folds


def _prepare_group_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray[Any, Any], pd.Series, dict[object, int]]:
    df = pd.read_excel(ROOT / args.excel, sheet_name=args.sheet_name, engine="openpyxl").copy()
    expected = set(_converter.STACK_COLS + _converter.COND_COLS + [_converter.TIME_COL, _converter.LABEL_COL])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in Excel: {sorted(missing)}")
    if args.split_mode == "segment":
        df["__group_key__"] = _converter._derive_segment_group_keys(
            df,
            gap_seconds=args.segment_gap_seconds,
            block_seconds=args.segment_block_seconds,
            label_boundary=args.segment_label_boundary,
        )
    else:
        df["__group_key__"] = _converter._derive_group_keys(pd.Series(df[_converter.TIME_COL].to_numpy()), args.split_mode)
    groups = np.asarray(df["__group_key__"].unique())
    group_label_map = df.groupby("__group_key__")[_converter.LABEL_COL].first()
    row_counts = {group: int(count) for group, count in df.groupby("__group_key__").size().items()}
    return df, groups, group_label_map, row_counts


def _build_fold_npz(
    df_input: pd.DataFrame,
    group_label_map: pd.Series,
    row_counts: dict[object, int],
    fold: dict[str, np.ndarray[Any, Any]],
    output_npz: Path,
    args: argparse.Namespace,
) -> Path:
    df = df_input.copy()
    g_tr = np.asarray(fold["train"])
    g_va = np.asarray(fold["val"])
    g_te = np.asarray(fold["test"])
    g_tr_set = set(g_tr.tolist())
    g_va_set = set(g_va.tolist())
    g_te_set = set(g_te.tolist())

    df[_converter.STACK_COLS] = df[_converter.STACK_COLS].fillna(0)
    df[_converter.COND_COLS] = df[_converter.COND_COLS].fillna(0)
    train_df = df[df["__group_key__"].isin(list(g_tr_set))].copy()
    _, stack_mean, stack_std = _converter.normalize_columns(train_df, _converter.STACK_COLS)
    _, cond_mean, cond_std = _converter.normalize_columns(train_df, _converter.COND_COLS)

    def _apply_norm(frame: pd.DataFrame, cols: list[str], means: dict[str, float], stds: dict[str, float]) -> pd.DataFrame:
        frame = frame.copy()
        for col in cols:
            std = stds.get(col, 1.0) or 1.0
            frame[col] = (frame[col] - means.get(col, 0.0)) / std
        return frame

    df = _apply_norm(df, _converter.STACK_COLS, stack_mean, stack_std)
    df = _apply_norm(df, _converter.COND_COLS, cond_mean, cond_std)

    resolved_min_train_stride = int(args.min_train_stride if args.min_train_stride is not None else max(1, args.stride_train // 2))
    resolved_max_train_stride = int(args.max_train_stride if args.max_train_stride is not None else max(args.stride_train, args.stride_train * 2))
    train_group_labels = np.array([group_label_map[g] for g in g_tr])
    stride_by_label = (
        _converter._class_aware_stride_map(
            train_group_labels,
            base_stride=args.stride_train,
            min_stride=resolved_min_train_stride,
            max_stride=resolved_max_train_stride,
            power=args.class_stride_power,
        )
        if args.class_aware_train_stride
        else {}
    )

    def _windows(
        group_set: set[object],
        stride: int,
        stride_map: dict[int, int] | None = None,
    ) -> tuple[list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any]], list[int]]:
        windows: list[np.ndarray[Any, Any]] = []
        conds: list[np.ndarray[Any, Any]] = []
        labels: list[int] = []
        local_stride_map = stride_map or {}
        for group in sorted(group_set, key=str):
            group_df = df.loc[df["__group_key__"] == group].sort_index()
            label = int(group_df[_converter.LABEL_COL].iloc[0])
            group_stride = max(1, int(local_stride_map.get(label, stride)))
            stack_values = group_df[_converter.STACK_COLS].to_numpy(dtype=np.float32)
            cond_values = group_df[_converter.COND_COLS].to_numpy(dtype=np.float32)
            row_count = len(group_df)
            if row_count < args.window_size:
                stack_padded = np.pad(stack_values, ((0, args.window_size - row_count), (0, 0)), mode="edge")
                windows.append(stack_padded.T.astype(np.float32))
                conds.append(cond_values.mean(axis=0).astype(np.float32))
                labels.append(label)
            else:
                for start in range(0, row_count - args.window_size + 1, group_stride):
                    windows.append(stack_values[start : start + args.window_size].T.astype(np.float32))
                    conds.append(cond_values[start : start + args.window_size].mean(axis=0).astype(np.float32))
                    labels.append(label)
        return windows, conds, labels

    train_w, train_c, train_y = _windows(g_tr_set, args.stride_train, stride_by_label)
    val_w, val_c, val_y = _windows(g_va_set, args.stride_eval)
    test_w, test_c, test_y = _windows(g_te_set, args.stride_eval)
    x_op = np.asarray(train_w + val_w + test_w, dtype=np.float32)
    x_cond = np.asarray(train_c + val_c + test_c, dtype=np.float32)
    labels = np.asarray(train_y + val_y + test_y, dtype=np.int64)
    split = np.concatenate(
        [
            np.zeros(len(train_y), dtype=np.int64),
            np.ones(len(val_y), dtype=np.int64),
            np.full(len(test_y), 2, dtype=np.int64),
        ]
    )
    x_eis = _converter._build_eis_sequence(x_cond, eis_seq_len=args.eis_seq_len)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, x_op=x_op, x_eis=x_eis, x_cond=x_cond, labels=labels, split=split)

    group_labels = np.array([group_label_map[g] for g in list(g_tr) + list(g_va) + list(g_te)])
    quality = _converter._split_quality(
        row_counts=row_counts,
        group_label_map=group_label_map,
        g_tr=g_tr,
        g_va=g_va,
        g_te=g_te,
        window_size=args.window_size,
        stride_train=args.stride_train,
        stride_eval=args.stride_eval,
        labels=[str(int(label)) for label in sorted(np.unique(group_labels).tolist())],
        min_eval_class_windows=args.min_eval_class_windows,
        min_eval_class_groups=args.min_eval_class_groups,
    )
    summary = {
        "output_path": str(output_npz),
        "split_protocol": "single_seed_stratified_group_kfold",
        "seed": int(args.seed),
        "num_samples": int(labels.shape[0]),
        "train_size": int((split == 0).sum()),
        "val_size": int((split == 1).sum()),
        "test_size": int((split == 2).sum()),
        "split_label_counts": {
            "train": _label_count_dict(labels[split == 0]),
            "val": _label_count_dict(labels[split == 1]),
            "test": _label_count_dict(labels[split == 2]),
        },
        "source_meta": {
            "n_groups_train": int(len(g_tr)),
            "n_groups_val": int(len(g_va)),
            "n_groups_test": int(len(g_te)),
            "group_label_counts_train": _label_count_dict(np.array([group_label_map[g] for g in g_tr])),
            "group_label_counts_val": _label_count_dict(np.array([group_label_map[g] for g in g_va])),
            "group_label_counts_test": _label_count_dict(np.array([group_label_map[g] for g in g_te])),
            "split_quality": quality,
        },
    }
    output_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_npz


def _safe_float(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["ratio"]), str(row["model"])), []).append(row)
    metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_inference_ms"]
    out: list[dict[str, Any]] = []
    for (ratio, model), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "ratio": ratio,
            "model": model,
            "category": group[0].get("category", ""),
            "n_folds": len(group),
            "fold_values": ";".join(str(item.get("fold", "")) for item in group),
            "parameter_count": int(round(fmean([_safe_float(item.get("parameter_count", 0)) for item in group]))),
        }
        for metric in metrics:
            values = [_safe_float(item.get(metric, 0.0)) for item in group]
            row[f"{metric}_mean"] = f"{fmean(values):.6f}"
            row[f"{metric}_std"] = f"{stdev(values):.6f}" if len(values) > 1 else "0.000000"
            row[f"{metric}_min"] = f"{min(values):.6f}"
            row[f"{metric}_max"] = f"{max(values):.6f}"
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official single-seed multi-fold experiments")
    parser.add_argument("--excel", default="data/raw/水淹和膜干故障测试数据_补充特征汇总.xlsx")
    parser.add_argument("--sheet-name", default="Sheet1")
    parser.add_argument("--models", nargs="+", default=["proposed", "logreg", "random_forest", "mlp", "cnn1d", "transformer", "itransformer"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output-root", default="results/codex_single_seed44_5fold_model_comparison")
    parser.add_argument("--data-root", default="data/processed/codex_single_seed44_5fold_model_comparison")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride-train", type=int, default=16)
    parser.add_argument("--stride-eval", type=int, default=32)
    parser.add_argument("--eis-seq-len", type=int, default=128)
    parser.add_argument("--split-mode", default="segment")
    parser.add_argument("--segment-gap-seconds", type=float, default=600.0)
    parser.add_argument("--segment-block-seconds", type=float, default=300.0)
    parser.add_argument("--segment-label-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--refit-trainval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-epochs-before-stop", type=int, default=20)
    parser.add_argument("--val-metric-smoothing", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--class-aware-train-stride", action="store_true")
    parser.add_argument("--min-train-stride", type=int, default=None)
    parser.add_argument("--max-train-stride", type=int, default=None)
    parser.add_argument("--class-stride-power", type=float, default=1.0)
    parser.add_argument("--class-weighting", default="sqrt_balanced")
    parser.add_argument("--proposed-config", default="configs/rbf_kanfusion.yaml")
    parser.add_argument("--checkpoint-selection", choices=["best_val", "best-val", "last"], default="best_val")
    parser.add_argument("--min-eval-class-windows", type=int, default=5)
    parser.add_argument("--min-eval-class-groups", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    data_root = ROOT / args.data_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    df, groups, group_label_map, row_counts = _prepare_group_frame(args)
    group_labels = np.array([group_label_map[g] for g in groups], dtype=np.int64)
    folds = _make_stratified_group_folds(groups, group_labels, args.folds, args.seed, args.val_size)

    rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        fold_key = f"fold_{fold_index}"
        npz_path = data_root / fold_key / f"official_self_stack_impedance_eis_w{args.window_size}_8_2.npz"
        _build_fold_npz(df, group_label_map, row_counts, fold, npz_path, args)
        for model_key in args.models:
            run_dir = output_root / fold_key / "8_2" / model_key
            print(f"\n=== fold={fold_index}/{args.folds} | model={model_key} | seed={args.seed} ===", flush=True)
            if model_key == "proposed":
                metrics_path = _baseline.run_proposed_model(
                    npz_path,
                    run_dir,
                    ROOT / args.proposed_config,
                    args.epochs,
                    args.patience,
                    args.min_delta,
                    args.batch_size,
                    args.refit_trainval,
                    args.min_epochs_before_stop,
                    args.val_metric_smoothing,
                    args.class_weighting,
                    args.seed,
                    args.lr,
                    args.weight_decay,
                    args.checkpoint_selection,
                )
            elif model_key in {"logreg", "svm", "random_forest"}:
                metrics_path = _baseline.run_ml_baseline(model_key, npz_path, run_dir, args.seed, args.rf_estimators, args.refit_trainval)
            else:
                metrics_path = _baseline.run_torch_baseline(
                    model_key,
                    npz_path,
                    run_dir,
                    args.epochs,
                    args.patience,
                    args.min_delta,
                    args.batch_size,
                    args.lr,
                    args.weight_decay,
                    args.seed,
                    args.hidden_dim,
                    args.d_model,
                    args.num_layers,
                    args.dropout,
                    args.refit_trainval,
                    args.min_epochs_before_stop,
                    args.val_metric_smoothing,
                    args.class_weighting,
                )
            row = _baseline._metric_row("8_2", model_key, metrics_path)
            row["fold"] = fold_index
            rows.append(row)
            _baseline._print_metric_row(row)

    detail_fields = [
        "fold",
        "ratio",
        "model",
        "category",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_inference_ms",
        "parameter_count",
        "metrics_path",
    ]
    _write_rows(output_root / "fold_test_summary.csv", rows, detail_fields)
    aggregate_fields = [
        "ratio",
        "model",
        "category",
        "n_folds",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_accuracy_min",
        "test_accuracy_max",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "test_macro_f1_min",
        "test_macro_f1_max",
        "test_weighted_f1_mean",
        "test_weighted_f1_std",
        "test_weighted_f1_min",
        "test_weighted_f1_max",
        "test_inference_ms_mean",
        "test_inference_ms_std",
        "test_inference_ms_min",
        "test_inference_ms_max",
        "parameter_count",
        "fold_values",
    ]
    aggregated = _aggregate(rows)
    _write_rows(output_root / "single_seed_fold_summary.csv", aggregated, aggregate_fields)
    ranked = sorted(aggregated, key=lambda row: -_safe_float(row["test_macro_f1_mean"]))
    _write_rows(output_root / "ranked_single_seed_fold_summary.csv", ranked, aggregate_fields)
    print(f"\nWrote detail: {output_root / 'fold_test_summary.csv'}", flush=True)
    print(f"Wrote summary: {output_root / 'single_seed_fold_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
