"""Self-measured PEMFC dataset loader for 测试数据.xlsx.

Loads the self-measured Excel data with explicit column mapping:
    x_op  = 216 single-cell voltages (单片1..单片216)
    cond  = 9 impedance/EIS statistics + 3 stack variables (12 dims)
    label = 类型

Implements group-based train/val/test split by 测试时间 to prevent
data leakage (each time group shares a single label).

Typical usage::

    from src.datasets.self_dataset import build_self_datasets

    train_ds, val_ds, test_ds, meta = build_self_datasets(
        excel_path="data/processed/测试数据.xlsx",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from .csv_dataset import CSVFuelCellDataset
from .feature_engineering import normalize_columns

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Column definitions for 测试数据.xlsx Sheet1
# ------------------------------------------------------------------

TIME_COL = "测试时间"
LABEL_COL = "类型"

# 9 impedance / EIS statistic features
EIS_COLS: List[str] = [
    "总阻抗", "平均阻抗", "最高阻抗", "次高阻抗",
    "最低阻抗", "次低阻抗", "标准差",
    "EIS电阻实部", "EIS电阻虚部",
]

# 216 single-cell voltages → operational time-series channels
CELL_VOLTAGE_COLS: List[str] = [f"单片{i}" for i in range(1, 217)]

# 3 stack-level variables
STACK_COLS: List[str] = ["电堆总电压", "电堆总电流", "电堆功率"]

# Derived branch mappings
OP_COLS: List[str] = CELL_VOLTAGE_COLS          # x_op: 216 channels
COND_COLS: List[str] = EIS_COLS + STACK_COLS     # cond: 12 dimensions


# ------------------------------------------------------------------
# Group-stratified splitting
# ------------------------------------------------------------------

def _derive_group_keys(
    time_series: pd.Series,
    mode: str,
) -> pd.Series:
    ts = pd.to_datetime(time_series)
    if mode == "timestamp":
        return ts.astype(str)
    if mode == "date":
        return ts.dt.date.astype(str)
    if mode == "hour":
        return ts.dt.floor("H").astype(str)
    if mode == "session30m":
        return ts.dt.floor("30min").astype(str)
    if mode == "session10m":
        return ts.dt.floor("10min").astype(str)
    raise ValueError(f"Unsupported self split mode: {mode}")


def _derive_segment_group_keys(
    frame: pd.DataFrame,
    *,
    gap_seconds: float = 600.0,
    block_seconds: float = 600.0,
    label_boundary: bool = True,
) -> pd.Series:
    """Derive local-continuity segment IDs from one combined Excel file.

    A new segment starts when the sorted timestamp gap is larger than
    ``gap_seconds``.  By default, a label change also starts a new segment so
    sliding windows never mix fault classes.  Long continuous segments are then
    split into fixed-duration blocks so group-level train/val/test splitting is
    still possible when a class appears in only one long run.
    """
    ts = pd.Series(pd.to_datetime(frame[TIME_COL], errors="coerce"), index=frame.index)
    if ts.isna().any():
        bad_rows = np.flatnonzero(ts.isna().to_numpy())[:5].tolist()
        raise ValueError(f"Invalid {TIME_COL} values at rows: {bad_rows}")

    ordered = pd.DataFrame({
        "__row__": np.arange(len(frame)),
        "__ts__": ts.to_numpy(),
        "__label__": frame[LABEL_COL].to_numpy(),
    }).sort_values(["__ts__", "__row__"], kind="mergesort")

    gaps = ordered["__ts__"].diff().dt.total_seconds()
    new_segment = gaps.isna() | (gaps > gap_seconds)
    if label_boundary:
        new_segment = new_segment | (ordered["__label__"] != ordered["__label__"].shift())

    segment_ids = new_segment.cumsum().astype(int) - 1
    if block_seconds > 0:
        segment_start_ts = ordered.groupby(segment_ids)["__ts__"].transform("first")
        elapsed_seconds = (ordered["__ts__"] - segment_start_ts).dt.total_seconds()
        block_ids = np.floor(elapsed_seconds / block_seconds).astype(int)
    else:
        block_ids = pd.Series(np.zeros(len(ordered), dtype=int), index=ordered.index)

    segment_keys = pd.Series(
        [
            f"segment_{segment_id:04d}_block_{block_id:04d}"
            for segment_id, block_id in zip(segment_ids, block_ids)
        ],
        index=ordered["__row__"].to_numpy(),
        dtype="object",
    )
    return segment_keys.sort_index().reset_index(drop=True)


def _group_split(
    groups: NDArray[np.object_],
    group_labels: NDArray[np.int64],
    test_size: float,
    val_size: float,
    random_state: int,
    strategy: str = "two_stage",
) -> Tuple[NDArray[np.object_], NDArray[np.object_], NDArray[np.object_]]:
    """Split unique group IDs into train/val/test, stratified by label.

    Parameters
    ----------
    groups : array of group identifiers (e.g. unique 测试时间 values)
    group_labels : array of integer labels, one per group
    test_size : fraction of groups for test
    val_size : fraction of remaining groups for validation
    random_state : random seed

    Returns
    -------
    g_train, g_val, g_test : arrays of group identifiers
    """
    normalized_strategy = str(strategy).lower()
    global_val_size = float(val_size) * max(0.0, 1.0 - float(test_size))
    if normalized_strategy == "holdout_first":
        holdout_size = float(test_size) + global_val_size
        if not 0.0 < holdout_size < 1.0:
            raise ValueError("holdout_first requires 0 < test_size + val_size * (1 - test_size) < 1")
        g_tr, g_holdout = train_test_split(
            groups,
            test_size=holdout_size,
            stratify=group_labels,
            random_state=random_state,
        )
        label_map = dict(zip(groups, group_labels))
        holdout_labels = np.array([label_map[g] for g in g_holdout])
        test_fraction_within_holdout = float(test_size) / holdout_size
        g_va, g_te = train_test_split(
            g_holdout,
            test_size=test_fraction_within_holdout,
            stratify=holdout_labels,
            random_state=random_state,
        )
        return np.asarray(g_tr), np.asarray(g_va), np.asarray(g_te)

    if normalized_strategy == "three_way":
        rng = np.random.default_rng(int(random_state))
        train_groups: List[object] = []
        val_groups: List[object] = []
        test_groups: List[object] = []
        for label in sorted(np.unique(group_labels).tolist()):
            class_groups = groups[group_labels == label].copy()
            rng.shuffle(class_groups)
            n_groups = int(len(class_groups))
            if n_groups == 1:
                train_groups.extend(class_groups.tolist())
                continue
            n_test = int(round(n_groups * float(test_size)))
            n_val = int(round(n_groups * global_val_size))
            if float(test_size) > 0 and n_groups >= 2:
                n_test = max(1, n_test)
            if global_val_size > 0 and n_groups >= 3:
                n_val = max(1, n_val)
            if n_test + n_val >= n_groups:
                overflow = n_test + n_val - n_groups + 1
                reduce_val = min(n_val, overflow)
                n_val -= reduce_val
                overflow -= reduce_val
                if overflow > 0:
                    n_test = max(0, n_test - overflow)
            test_groups.extend(class_groups[:n_test].tolist())
            val_groups.extend(class_groups[n_test : n_test + n_val].tolist())
            train_groups.extend(class_groups[n_test + n_val :].tolist())
        if not train_groups or not val_groups or not test_groups:
            raise ValueError(
                "three_way split produced an empty split; reduce val/test ratio or use strategy='two_stage' for this dataset"
            )
        return np.asarray(train_groups), np.asarray(val_groups), np.asarray(test_groups)
    if normalized_strategy != "two_stage":
        raise ValueError(f"Unsupported group split strategy: {strategy}")

    g_tv, g_te = train_test_split(
        groups,
        test_size=test_size,
        stratify=group_labels,
        random_state=random_state,
    )
    label_map = dict(zip(groups, group_labels))
    tv_labels = np.array([label_map[g] for g in g_tv])
    g_tr, g_va = train_test_split(
        g_tv,
        test_size=val_size,
        stratify=tv_labels,
        random_state=random_state,
    )
    return np.asarray(g_tr), np.asarray(g_va), np.asarray(g_te)


# ------------------------------------------------------------------
# Windowing within a single time group
# ------------------------------------------------------------------

def _create_group_windows(
    group_df: pd.DataFrame,
    op_cols: List[str],
    cond_cols: List[str],
    label: int,
    window_size: int,
    stride: int,
) -> Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]], List[int]]:
    """Slide a window over rows of a single time group.

    Returns lists of (op_window [C, T], cond_vector [d], label).
    Short groups are padded with edge replication.
    """
    n = len(group_df)
    op_vals = np.asarray(group_df[op_cols].to_numpy(dtype=np.float32))      # [n, 216]
    cond_vals = np.asarray(group_df[cond_cols].to_numpy(dtype=np.float32))   # [n, 12]

    windows: List[NDArray[np.float32]] = []
    conds: List[NDArray[np.float32]] = []
    labels: List[int] = []

    if n < window_size:
        op_padded = np.pad(np.asarray(op_vals), ((0, window_size - n), (0, 0)), mode="edge")
        windows.append(op_padded.T)          # [216, window_size]
        conds.append(cond_vals.mean(axis=0))
        labels.append(label)
    else:
        for s in range(0, n - window_size + 1, stride):
            chunk_op = op_vals[s : s + window_size]
            chunk_cond = cond_vals[s : s + window_size]
            windows.append(chunk_op.T)       # [216, window_size]
            conds.append(chunk_cond.mean(axis=0))
            labels.append(label)

    return windows, conds, labels


# ------------------------------------------------------------------
# Public builder
# ------------------------------------------------------------------

def build_self_datasets(
    excel_path: str | Path,
    *,
    sheet_name: str = "Sheet1",
    window_size: int = 32,
    stride_train: int = 8,
    stride_eval: int = 16,
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42,
    augment_train: bool = True,
    label_names: Optional[List[str]] = None,
    split_mode: str = "timestamp",
    segment_gap_seconds: float = 600.0,
    segment_block_seconds: float = 600.0,
    segment_label_boundary: bool = True,
    feature_subset: str = "full",
    group_split_strategy: str = "two_stage",
) -> Tuple[CSVFuelCellDataset, CSVFuelCellDataset, CSVFuelCellDataset, Dict[str, Any]]:
    """Load self-measured Excel → group split → window → datasets.

    Parameters
    ----------
    excel_path : path to 测试数据.xlsx
    sheet_name : Excel sheet name (default "Sheet1")
    window_size : sliding window length in rows (time steps)
    stride_train / stride_eval : window stride for train / eval splits
    test_size : fraction of time groups reserved for test
    val_size : fraction of remaining groups reserved for validation
    random_state : random seed for splitting
    augment_train : enable online augmentation for training set
    label_names : optional display names for classes

    Returns
    -------
    train_ds, val_ds, test_ds : CSVFuelCellDataset instances
    meta : dict with dataset metadata
    """
    logger.info("Loading self-measured data from %s [%s]", excel_path, sheet_name)
    df = pd.read_excel(Path(excel_path), sheet_name=sheet_name, engine="openpyxl")
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Validate expected columns
    expected = set(OP_COLS + COND_COLS + [TIME_COL, LABEL_COL])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in Excel: {sorted(missing)}")

    # Derive group key according to configured split strictness
    df = df.copy()
    if split_mode == "segment":
        df["__group_key__"] = _derive_segment_group_keys(
            df,
            gap_seconds=segment_gap_seconds,
            block_seconds=segment_block_seconds,
            label_boundary=segment_label_boundary,
        )
    else:
        time_values = pd.Series(df[TIME_COL].to_numpy())
        df["__group_key__"] = _derive_group_keys(time_values, split_mode)

    # Extract unique groups and their labels
    groups = np.asarray(df["__group_key__"].unique())
    group_label_map = df.groupby("__group_key__")[LABEL_COL].first()
    group_labels = np.array([group_label_map[g] for g in groups])

    logger.info(
        "Found %d unique time groups, label distribution: %s",
        len(groups),
        dict(zip(*np.unique(group_labels, return_counts=True))),
    )

    # Stratified group split
    g_tr, g_va, g_te = _group_split(
        groups, group_labels, test_size, val_size, random_state, strategy=group_split_strategy,
    )
    logger.info(
        "Group split: %d train / %d val / %d test",
        len(g_tr), len(g_va), len(g_te),
    )

    g_tr_set = set(g_tr.tolist())
    g_va_set = set(g_va.tolist())
    g_te_set = set(g_te.tolist())

    # Fill NaN after split planning
    df[OP_COLS] = df[OP_COLS].fillna(0)
    df[COND_COLS] = df[COND_COLS].fillna(0)

    # Fit normalization statistics on train groups only, then apply to all
    train_df = cast(pd.DataFrame, df[df["__group_key__"].isin(list(g_tr_set))].copy())
    _, op_mean, op_std = normalize_columns(train_df, OP_COLS)
    _, cond_mean, cond_std = normalize_columns(train_df, COND_COLS)

    def _apply_norm(frame: pd.DataFrame, cols: List[str], means: Dict[str, float], stds: Dict[str, float]) -> pd.DataFrame:
        frame = frame.copy()
        for c in cols:
            s = stds.get(c, 1.0) or 1.0
            frame[c] = (frame[c] - means.get(c, 0.0)) / s
        return frame

    df = _apply_norm(df, OP_COLS, op_mean, op_std)
    df = _apply_norm(df, COND_COLS, cond_mean, cond_std)

    group_to_domain: Dict[str, int] = {}

    def _build_split(
        group_set: set[object], stride: int,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64], NDArray[np.int64], List[str]]:
        all_w: List[NDArray[np.float32]] = []
        all_c: List[NDArray[np.float32]] = []
        all_l: List[int] = []
        all_d: List[int] = []
        all_ids: List[str] = []
        for g in sorted(group_set, key=str):
            gdf = cast(pd.DataFrame, df.loc[df["__group_key__"] == g].sort_index())
            label = int(cast(pd.Series, gdf[LABEL_COL]).iloc[0])
            w, c, l = _create_group_windows(
                gdf, OP_COLS, COND_COLS, label, window_size, stride,
            )
            all_w.extend(w)
            all_c.extend(c)
            all_l.extend(l)
            g_key = str(g)
            if g_key not in group_to_domain:
                group_to_domain[g_key] = len(group_to_domain)
            all_d.extend([group_to_domain[g_key]] * len(l))
            all_ids.extend([f"{g}__w{i}" for i in range(len(l))])
        return (
            np.array(all_w, dtype=np.float32),
            np.array(all_c, dtype=np.float32),
            np.array(all_l, dtype=np.int64),
            np.array(all_d, dtype=np.int64),
            all_ids,
        )

    X_tr, Xc_tr, y_tr, d_tr, id_tr = _build_split(g_tr_set, stride_train)
    X_va, Xc_va, y_va, d_va, id_va = _build_split(g_va_set, stride_eval)
    X_te, Xc_te, y_te, d_te, id_te = _build_split(g_te_set, stride_eval)

    def _apply_subset(
        x_op: NDArray[np.float32],
        x_cond: NDArray[np.float32],
        subset: str,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        x_op = x_op.copy()
        x_cond = x_cond.copy()
        if subset == "full":
            return x_op, x_cond
        if subset == "op_only":
            x_cond[:] = 0
            return x_op, x_cond
        if subset == "cond_only":
            x_op[:] = 0
            return x_op, x_cond
        if subset == "eis_stat_only":
            x_op[:] = 0
            x_cond[:, 9:] = 0
            return x_op, x_cond
        if subset == "stack_only":
            x_op[:] = 0
            x_cond[:, :9] = 0
            return x_op, x_cond
        raise ValueError(f"Unsupported feature_subset: {subset}")

    X_tr, Xc_tr = _apply_subset(X_tr, Xc_tr, feature_subset)
    X_va, Xc_va = _apply_subset(X_va, Xc_va, feature_subset)
    X_te, Xc_te = _apply_subset(X_te, Xc_te, feature_subset)

    train_ds = CSVFuelCellDataset(X_tr, Xc_tr, y_tr, domains=d_tr, sample_ids=id_tr, augment=augment_train)
    val_ds = CSVFuelCellDataset(X_va, Xc_va, y_va, domains=d_va, sample_ids=id_va)
    test_ds = CSVFuelCellDataset(X_te, Xc_te, y_te, domains=d_te, sample_ids=id_te)

    all_labels = np.concatenate([y_tr, y_va, y_te])
    num_classes = len(np.unique(all_labels))
    _label_names = label_names or [str(i) for i in sorted(np.unique(all_labels))]

    meta: Dict[str, Any] = {
        "label_names": _label_names,
        "num_classes": num_classes,
        "op_cols": OP_COLS,
        "cond_cols": COND_COLS,
        "n_op_channels": len(OP_COLS),
        "n_cond_dims": len(COND_COLS),
        "op_mean": op_mean,
        "op_std": op_std,
        "samples_per_class_train": np.bincount(y_tr, minlength=num_classes).tolist(),
        "window_size": window_size,
        "train_size": len(y_tr),
        "val_size": len(y_va),
        "test_size": len(y_te),
        "n_groups": len(groups),
        "n_groups_train": len(g_tr),
        "n_groups_val": len(g_va),
        "n_groups_test": len(g_te),
        "split_strategy": f"group_stratified_by_{split_mode}",
        "group_split_strategy": group_split_strategy,
        "group_values_train": [str(g) for g in sorted(g_tr_set)],
        "group_values_val": [str(g) for g in sorted(g_va_set)],
        "group_values_test": [str(g) for g in sorted(g_te_set)],
        "group_mode": split_mode,
        "segment_gap_seconds": segment_gap_seconds if split_mode == "segment" else None,
        "segment_block_seconds": segment_block_seconds if split_mode == "segment" else None,
        "segment_label_boundary": segment_label_boundary if split_mode == "segment" else None,
        "feature_subset": feature_subset,
    }

    logger.info(
        "Self dataset built: %d train / %d val / %d test windows, "
        "%d op channels, %d cond dims, %d classes",
        meta["train_size"], meta["val_size"], meta["test_size"],
        meta["n_op_channels"], meta["n_cond_dims"], meta["num_classes"],
    )
    return train_ds, val_ds, test_ds, meta
