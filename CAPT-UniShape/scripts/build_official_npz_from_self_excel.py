"""Convert self-measured Excel data to the NPZ format used by official models.

The source Excel (`data/processed/测试数据.xlsx`) contains 216 single-cell
voltages, nine impedance/EIS statistical features, three stack-level variables,
timestamps and labels.  This converter reuses the existing group-stratified
Excel loader to avoid leakage, then builds the three official model inputs:

* x_op   [N, 216, window_size]
* x_eis  [N, 4, eis_seq_len]
* x_cond [N, 12]
* labels [N]
* split  [N] where 0=train, 1=val, 2=test

The EIS sequence is a deterministic research-ready representation derived from
the nine impedance statistics: interpolated statistic curve, first difference,
cumulative shape and normalized frequency coordinate.  If raw full-frequency
EIS spectra become available later, replace this builder with the raw spectral
channels while keeping the same NPZ keys.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_feature_engineering = importlib.import_module("src.datasets.feature_engineering")
_self_dataset = importlib.import_module("src.datasets.self_dataset")
normalize_columns = _feature_engineering.normalize_columns
COND_COLS = _self_dataset.COND_COLS
EIS_COLS = _self_dataset.EIS_COLS
LABEL_COL = _self_dataset.LABEL_COL
OP_COLS = _self_dataset.OP_COLS
STACK_COLS = _self_dataset.STACK_COLS
TIME_COL = _self_dataset.TIME_COL
_derive_group_keys = _self_dataset._derive_group_keys
_derive_segment_group_keys = _self_dataset._derive_segment_group_keys
_group_split = _self_dataset._group_split
build_self_datasets = _self_dataset.build_self_datasets


def _dataset_arrays(dataset: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    x_op = dataset.x_op.detach().cpu().numpy().astype(np.float32)
    x_cond = dataset.x_cond.detach().cpu().numpy().astype(np.float32)
    labels = dataset.labels.detach().cpu().numpy().astype(np.int64)
    return x_op, x_cond, labels


def _build_eis_sequence(x_cond: np.ndarray[Any, Any], eis_seq_len: int) -> np.ndarray[Any, Any]:
    """Create four EIS-like ordered channels from nine impedance statistics."""
    eis_stats = x_cond[:, : len(EIS_COLS)].astype(np.float32)
    source_grid = np.linspace(0.0, 1.0, eis_stats.shape[1], dtype=np.float32)
    target_grid = np.linspace(0.0, 1.0, int(eis_seq_len), dtype=np.float32)
    sequences: list[np.ndarray[Any, Any]] = []
    for row in eis_stats:
        curve = np.interp(target_grid, source_grid, row).astype(np.float32)
        diff = np.gradient(curve).astype(np.float32)
        centered = curve - float(curve.mean())
        cumulative = np.cumsum(centered).astype(np.float32)
        denom = float(np.max(np.abs(cumulative))) or 1.0
        cumulative = cumulative / denom
        freq_axis = target_grid.astype(np.float32)
        sequences.append(np.stack([curve, diff, cumulative, freq_axis], axis=0))
    return np.stack(sequences, axis=0).astype(np.float32)


def _label_count_dict(values: np.ndarray[Any, Any]) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def _class_aware_stride_map(
    labels: np.ndarray[Any, Any],
    base_stride: int,
    min_stride: int,
    max_stride: int,
    power: float,
) -> dict[int, int]:
    counts = {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    if not counts:
        return {}
    min_count = min(counts.values())
    max_count = max(counts.values())
    stride_map: dict[int, int] = {}
    for label, count in counts.items():
        if max_count == min_count:
            stride = int(base_stride)
        else:
            normalized = (float(count) - float(min_count)) / (float(max_count) - float(min_count))
            stride = int(round(float(min_stride) + (normalized ** float(power)) * float(max_stride - min_stride)))
        stride_map[label] = max(int(min_stride), min(int(max_stride), stride))
    return stride_map


def _build_stack_windows(
    excel_path: str | Path,
    sheet_name: str,
    window_size: int,
    stride_train: int,
    stride_eval: int,
    split_mode: str,
    segment_gap_seconds: float,
    segment_block_seconds: float,
    segment_label_boundary: bool,
    random_state: int,
    test_size: float,
    val_size: float,
    class_aware_train_stride: bool,
    min_train_stride: int,
    max_train_stride: int,
    class_stride_power: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, object]]:
    df = pd.read_excel(Path(excel_path), sheet_name=sheet_name, engine="openpyxl")
    expected = set(STACK_COLS + COND_COLS + [TIME_COL, LABEL_COL])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in Excel: {sorted(missing)}")
    df = df.copy()
    if split_mode == "segment":
        df["__group_key__"] = _derive_segment_group_keys(
            df,
            gap_seconds=segment_gap_seconds,
            block_seconds=segment_block_seconds,
            label_boundary=segment_label_boundary,
        )
    else:
        df["__group_key__"] = _derive_group_keys(pd.Series(df[TIME_COL].to_numpy()), split_mode)
    groups = np.asarray(df["__group_key__"].unique())
    group_label_map = df.groupby("__group_key__")[LABEL_COL].first()
    group_labels = np.array([group_label_map[g] for g in groups])
    g_tr, g_va, g_te = _group_split(groups, group_labels, test_size, val_size, random_state)
    g_tr_set = set(g_tr.tolist())
    g_va_set = set(g_va.tolist())
    g_te_set = set(g_te.tolist())

    df[STACK_COLS] = df[STACK_COLS].fillna(0)
    df[COND_COLS] = df[COND_COLS].fillna(0)
    train_df = df[df["__group_key__"].isin(list(g_tr_set))].copy()
    _, stack_mean, stack_std = normalize_columns(train_df, STACK_COLS)
    _, cond_mean, cond_std = normalize_columns(train_df, COND_COLS)

    def _apply_norm(frame: pd.DataFrame, cols: list[str], means: dict[str, float], stds: dict[str, float]) -> pd.DataFrame:
        frame = frame.copy()
        for col in cols:
            std = stds.get(col, 1.0) or 1.0
            frame[col] = (frame[col] - means.get(col, 0.0)) / std
        return frame

    df = _apply_norm(df, STACK_COLS, stack_mean, stack_std)
    df = _apply_norm(df, COND_COLS, cond_mean, cond_std)

    train_group_labels = np.array([group_label_map[g] for g in g_tr])
    train_stride_by_label = (
        _class_aware_stride_map(
            train_group_labels,
            base_stride=stride_train,
            min_stride=min_train_stride,
            max_stride=max_train_stride,
            power=class_stride_power,
        )
        if class_aware_train_stride
        else {}
    )

    def _windows(
        group_set: set[object],
        stride: int,
        stride_by_label: dict[int, int] | None = None,
    ) -> tuple[list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any]], list[int]]:
        windows: list[np.ndarray[Any, Any]] = []
        conds: list[np.ndarray[Any, Any]] = []
        labels: list[int] = []
        stride_map = stride_by_label or {}
        for group in sorted(group_set, key=str):
            group_df = df.loc[df["__group_key__"] == group].sort_index()
            label = int(group_df[LABEL_COL].iloc[0])
            group_stride = max(1, int(stride_map.get(label, stride)))
            stack_values = group_df[STACK_COLS].to_numpy(dtype=np.float32)
            cond_values = group_df[COND_COLS].to_numpy(dtype=np.float32)
            row_count = len(group_df)
            if row_count < window_size:
                stack_padded = np.pad(stack_values, ((0, window_size - row_count), (0, 0)), mode="edge")
                windows.append(stack_padded.T.astype(np.float32))
                conds.append(cond_values.mean(axis=0).astype(np.float32))
                labels.append(label)
            else:
                for start in range(0, row_count - window_size + 1, group_stride):
                    windows.append(stack_values[start : start + window_size].T.astype(np.float32))
                    conds.append(cond_values[start : start + window_size].mean(axis=0).astype(np.float32))
                    labels.append(label)
        return windows, conds, labels

    train_w, train_c, train_y = _windows(g_tr_set, stride_train, train_stride_by_label)
    val_w, val_c, val_y = _windows(g_va_set, stride_eval)
    test_w, test_c, test_y = _windows(g_te_set, stride_eval)
    x_op = np.asarray(train_w + val_w + test_w, dtype=np.float32)
    x_cond = np.asarray(train_c + val_c + test_c, dtype=np.float32)
    labels = np.asarray(train_y + val_y + test_y, dtype=np.int64)
    split = np.concatenate([
        np.zeros(len(train_y), dtype=np.int64),
        np.ones(len(val_y), dtype=np.int64),
        np.full(len(test_y), 2, dtype=np.int64),
    ])
    meta: dict[str, object] = {
        "n_groups": len(groups),
        "n_groups_train": len(g_tr),
        "n_groups_val": len(g_va),
        "n_groups_test": len(g_te),
        "group_label_counts_train": _label_count_dict(np.array([group_label_map[g] for g in g_tr])),
        "group_label_counts_val": _label_count_dict(np.array([group_label_map[g] for g in g_va])),
        "group_label_counts_test": _label_count_dict(np.array([group_label_map[g] for g in g_te])),
        "class_aware_train_stride": class_aware_train_stride,
        "train_stride_by_label": {str(label): int(stride) for label, stride in train_stride_by_label.items()},
        "op_cols": STACK_COLS,
        "cond_cols": COND_COLS,
        "test_size_ratio": test_size,
        "val_size_ratio_within_train_pool": val_size,
    }
    return x_op, x_cond, labels, split, meta


def build_npz(
    excel_path: str | Path,
    output_path: str | Path,
    sheet_name: str = "Sheet1",
    window_size: int = 256,
    stride_train: int = 64,
    stride_eval: int = 128,
    eis_seq_len: int = 128,
    split_mode: str = "session10m",
    segment_gap_seconds: float = 600.0,
    segment_block_seconds: float = 600.0,
    segment_label_boundary: bool = True,
    feature_subset: str = "full",
    random_state: int = 42,
    op_source: str = "stack",
    test_size: float = 0.20,
    val_size: float = 0.15,
    class_aware_train_stride: bool = False,
    min_train_stride: int | None = None,
    max_train_stride: int | None = None,
    class_stride_power: float = 1.0,
) -> dict[str, object]:
    resolved_min_train_stride = int(min_train_stride if min_train_stride is not None else max(1, stride_train // 2))
    resolved_max_train_stride = int(max_train_stride if max_train_stride is not None else max(stride_train, stride_train * 2))
    if resolved_min_train_stride <= 0 or resolved_max_train_stride <= 0:
        raise ValueError("min_train_stride and max_train_stride must be positive")
    if resolved_min_train_stride > resolved_max_train_stride:
        raise ValueError("min_train_stride must be <= max_train_stride")
    if op_source == "stack":
        x_op, x_cond, labels, split, meta = _build_stack_windows(
            excel_path=excel_path,
            sheet_name=sheet_name,
            window_size=window_size,
            stride_train=stride_train,
            stride_eval=stride_eval,
            split_mode=split_mode,
            segment_gap_seconds=segment_gap_seconds,
            segment_block_seconds=segment_block_seconds,
            segment_label_boundary=segment_label_boundary,
            random_state=random_state,
            test_size=test_size,
            val_size=val_size,
            class_aware_train_stride=class_aware_train_stride,
            min_train_stride=resolved_min_train_stride,
            max_train_stride=resolved_max_train_stride,
            class_stride_power=class_stride_power,
        )
    elif op_source == "cells":
        if class_aware_train_stride:
            raise NotImplementedError("class-aware train stride is currently implemented for --op-source stack")
        train_ds, val_ds, test_ds, meta = build_self_datasets(
            excel_path=excel_path,
            sheet_name=sheet_name,
            window_size=window_size,
            stride_train=stride_train,
            stride_eval=stride_eval,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            augment_train=False,
            split_mode=split_mode,
            segment_gap_seconds=segment_gap_seconds,
            segment_block_seconds=segment_block_seconds,
            segment_label_boundary=segment_label_boundary,
            feature_subset=feature_subset,
        )
        train_op, train_cond, train_y = _dataset_arrays(train_ds)
        val_op, val_cond, val_y = _dataset_arrays(val_ds)
        test_op, test_cond, test_y = _dataset_arrays(test_ds)
        x_op = np.concatenate([train_op, val_op, test_op], axis=0)
        x_cond = np.concatenate([train_cond, val_cond, test_cond], axis=0)
        labels = np.concatenate([train_y, val_y, test_y], axis=0)
        split = np.concatenate([
            np.zeros(len(train_y), dtype=np.int64),
            np.ones(len(val_y), dtype=np.int64),
            np.full(len(test_y), 2, dtype=np.int64),
        ])
    else:
        raise ValueError("op_source must be 'stack' or 'cells'")
    x_eis = _build_eis_sequence(x_cond, eis_seq_len=eis_seq_len)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x_op=x_op,
        x_eis=x_eis,
        x_cond=x_cond,
        labels=labels,
        split=split,
    )
    summary: dict[str, object] = {
        "output_path": str(output),
        "num_samples": int(labels.shape[0]),
        "train_size": int((split == 0).sum()),
        "val_size": int((split == 1).sum()),
        "test_size": int((split == 2).sum()),
        "x_op_shape": list(x_op.shape),
        "x_eis_shape": list(x_eis.shape),
        "x_cond_shape": list(x_cond.shape),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "split_label_counts": {
            "train": _label_count_dict(labels[split == 0]),
            "val": _label_count_dict(labels[split == 1]),
            "test": _label_count_dict(labels[split == 2]),
        },
        "split_mode": split_mode,
        "segment_gap_seconds": segment_gap_seconds if split_mode == "segment" else None,
        "segment_block_seconds": segment_block_seconds if split_mode == "segment" else None,
        "segment_label_boundary": segment_label_boundary if split_mode == "segment" else None,
        "class_aware_train_stride": class_aware_train_stride,
        "min_train_stride": resolved_min_train_stride,
        "max_train_stride": resolved_max_train_stride,
        "class_stride_power": class_stride_power,
        "feature_subset": feature_subset,
        "op_source": op_source,
        "source_meta": {
            "n_groups": meta.get("n_groups"),
            "n_groups_train": meta.get("n_groups_train"),
            "n_groups_val": meta.get("n_groups_val"),
            "n_groups_test": meta.get("n_groups_test"),
            "group_label_counts_train": meta.get("group_label_counts_train"),
            "group_label_counts_val": meta.get("group_label_counts_val"),
            "group_label_counts_test": meta.get("group_label_counts_test"),
            "train_stride_by_label": meta.get("train_stride_by_label"),
            "test_size_ratio": test_size,
            "val_size_ratio_within_train_pool": val_size,
        },
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official CAPT-UniShape NPZ from 测试数据.xlsx")
    parser.add_argument("--excel", default="data/processed/测试数据.xlsx")
    parser.add_argument("--output", default="data/processed/official_self_multisource.npz")
    parser.add_argument("--sheet-name", default="Sheet1")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride-train", type=int, default=64)
    parser.add_argument("--stride-eval", type=int, default=128)
    parser.add_argument("--eis-seq-len", type=int, default=128)
    parser.add_argument("--split-mode", default="segment")
    parser.add_argument("--segment-gap-seconds", type=float, default=600.0)
    parser.add_argument("--segment-block-seconds", type=float, default=600.0)
    parser.add_argument("--segment-label-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-subset", default="full")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--op-source", choices=["stack", "cells"], default="stack")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test fraction, e.g. 0.2 for 8:2")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation fraction inside the non-test training pool for early stopping")
    parser.add_argument("--class-aware-train-stride", action="store_true", help="Use smaller train strides for minority classes and larger strides for majority classes")
    parser.add_argument("--min-train-stride", type=int, default=None, help="Minimum class-aware training stride; default=stride_train//2")
    parser.add_argument("--max-train-stride", type=int, default=None, help="Maximum class-aware training stride; default=stride_train*2")
    parser.add_argument("--class-stride-power", type=float, default=1.0, help="Power for class-count-to-stride scaling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_npz(
        excel_path=args.excel,
        output_path=args.output,
        sheet_name=args.sheet_name,
        window_size=args.window_size,
        stride_train=args.stride_train,
        stride_eval=args.stride_eval,
        eis_seq_len=args.eis_seq_len,
        split_mode=args.split_mode,
        segment_gap_seconds=args.segment_gap_seconds,
        segment_block_seconds=args.segment_block_seconds,
        segment_label_boundary=args.segment_label_boundary,
        feature_subset=args.feature_subset,
        random_state=args.random_state,
        op_source=args.op_source,
        test_size=args.test_size,
        val_size=args.val_size,
        class_aware_train_stride=args.class_aware_train_stride,
        min_train_stride=args.min_train_stride,
        max_train_stride=args.max_train_stride,
        class_stride_power=args.class_stride_power,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
