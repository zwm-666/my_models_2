"""CSV-based PEMFC dataset with feature engineering and online augmentation.

Typical usage::

    from src.datasets.csv_dataset import build_csv_datasets

    train_ds, val_ds, test_ds, meta = build_csv_datasets(
        csv_path="data/processed/data.csv",
        op_cols=["U_totV", "iA", ...],
        cond_cols=["i_write", ...],
        label_col="State",
        time_col="tsec",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .feature_engineering import engineer_features, normalize_columns


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class CSVFuelCellDataset(Dataset):
    """PyTorch Dataset backed by in-memory numpy arrays.

    Parameters
    ----------
    x_op : np.ndarray  ``(N, C, T)``
    x_cond : np.ndarray  ``(N, d_cond)``
    labels : np.ndarray  ``(N,)``
    augment : bool
        Online augmentation (noise, scaling, masking).
    """

    def __init__(
        self,
        x_op: np.ndarray,
        x_cond: np.ndarray,
        labels: np.ndarray,
        domains: Optional[np.ndarray] = None,
        sample_ids: Optional[List[str]] = None,
        augment: bool = False,
        noise_std: float = 0.05,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        mask_ratio: float = 0.1,
    ) -> None:
        self.x_op = torch.as_tensor(x_op, dtype=torch.float32)
        self.x_cond = torch.as_tensor(x_cond, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        n_samples = len(self.labels)
        if domains is None:
            self.domains = torch.zeros(n_samples, dtype=torch.long)
        else:
            self.domains = torch.as_tensor(domains, dtype=torch.long)
            if len(self.domains) != n_samples:
                raise ValueError("domains must have the same length as labels")
        if sample_ids is None:
            self.sample_ids = [str(i) for i in range(n_samples)]
        else:
            self.sample_ids = [str(sample_id) for sample_id in sample_ids]
            if len(self.sample_ids) != n_samples:
                raise ValueError("sample_ids must have the same length as labels")
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.x_op[idx].clone()
        if self.augment:
            x = self._augment(x)
        return {
            "x_op": x,
            "x_eis": torch.zeros(4, 32),
            "cond": self.x_cond[idx],
            "label": self.labels[idx].item(),
            "domain": int(self.domains[idx].item()),
            "sample_id": self.sample_ids[idx],
            "modality_mask": torch.tensor([1.0, 0.0]),
        }

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * self.noise_std
        if torch.rand(1).item() < 0.3:
            lo, hi = self.scale_range
            x = x * (lo + torch.rand(1).item() * (hi - lo))
        if torch.rand(1).item() < 0.3:
            t = x.shape[1]
            ml = max(1, int(t * self.mask_ratio))
            s = torch.randint(0, max(1, t - ml), (1,)).item()
            x[:, s:s + ml] = 0.0
        return x


# ------------------------------------------------------------------
# Windowing
# ------------------------------------------------------------------

def _create_windows(
    df: pd.DataFrame,
    op_cols: List[str],
    cond_cols: List[str],
    label_col: str,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.copy()
    df["_seg"] = (df[label_col] != df[label_col].shift()).cumsum()

    windows, conds, labels = [], [], []
    for _, seg in df.groupby("_seg"):
        n, lbl = len(seg), seg[label_col].iloc[0]
        if n < window_size // 4:
            continue
        if n < window_size:
            op = seg[op_cols].values
            op = np.pad(op, ((0, window_size - n), (0, 0)), mode="edge")
            windows.append(op.T)
            conds.append(seg[cond_cols].mean().values)
            labels.append(lbl)
            continue
        for s in range(0, n - window_size + 1, stride):
            c = seg.iloc[s:s + window_size]
            windows.append(c[op_cols].values.T)
            conds.append(c[cond_cols].mean().values)
            labels.append(lbl)

    return (np.array(windows, dtype=np.float32),
            np.array(conds, dtype=np.float32),
            np.array(labels, dtype=np.int64))


# ------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------

def _auto_detect_columns(
    df: pd.DataFrame,
    label_col: str,
    label_name_col: Optional[str],
) -> Tuple[List[str], List[str]]:
    """Infer op_cols and cond_cols from a CSV with no explicit column spec.

    Heuristic: exclude metadata/label columns, then split numeric columns
    by within-file variance (constant per file but varying globally → cond).
    """
    meta = {"cell_id", "source_file", "source_dir", label_col}
    if label_name_col:
        meta.add(label_name_col)

    numeric = [c for c in df.select_dtypes(include="number").columns if c not in meta]

    # Use source_file for grouping if available, else fall back to label segments
    if "source_file" in df.columns:
        group_col = "source_file"
    else:
        df = df.copy()
        df["_seg"] = (df[label_col] != df[label_col].shift()).cumsum()
        group_col = "_seg"

    global_std = df[numeric].std()
    per_group_std = df.groupby(group_col)[numeric].std().mean()

    cond_cols = [c for c in numeric
                 if global_std[c] > 1e-10 and per_group_std[c] < 1e-6]
    op_cols = [c for c in numeric
               if global_std[c] > 1e-10 and c not in cond_cols]
    return op_cols, cond_cols


def build_csv_datasets(
    csv_path: str | Path,
    *,
    op_cols: Optional[List[str]] = None,
    cond_cols: Optional[List[str]] = None,
    label_col: str = "State",
    time_col: str = "tsec",
    label_name_col: Optional[str] = "State_Label",
    window_size: int = 64,
    stride_train: int = 12,
    stride_eval: int = 32,
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42,
    augment_train: bool = True,
) -> Tuple[CSVFuelCellDataset, CSVFuelCellDataset, CSVFuelCellDataset, Dict[str, Any]]:
    """Load CSV → feature engineering → window → split → datasets.

    Parameters
    ----------
    csv_path : path
        Path to the CSV file.
    op_cols : list of str or None
        Raw operational sensor column names.  When *None*, columns are
        auto-detected from the CSV.
    cond_cols : list of str or None
        Condition set-point column names.  When *None*, auto-detected.
    label_col : str
        Integer label column.
    time_col : str
        Time column for sorting.
    label_name_col : str or None
        Optional string label column for display names.
    window_size, stride_train, stride_eval : int
        Sliding window parameters.
    test_size, val_size : float
        Split fractions.
    random_state : int
        Random seed.
    augment_train : bool
        Enable online augmentation.

    Returns
    -------
    train_ds, val_ds, test_ds, meta
    """
    df = pd.read_csv(Path(csv_path))

    # Sort by time
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    # Auto-detect columns when not provided
    if op_cols is None or cond_cols is None:
        _op, _cond = _auto_detect_columns(df, label_col, label_name_col)
        op_cols = op_cols or _op
        cond_cols = cond_cols or _cond

    # Feature engineering
    df, final_op, final_cond = engineer_features(df, op_cols, cond_cols)

    # Label info
    if label_name_col and label_name_col in df.columns:
        label_names = sorted(df[label_name_col].unique())
    else:
        label_names = [str(i) for i in sorted(df[label_col].unique())]
    num_classes = df[label_col].nunique()

    # Normalise
    df, op_mean, op_std = normalize_columns(df, final_op)
    df, cond_mean, cond_std = normalize_columns(df, final_cond)

    # Windows
    X_op, X_cond, y = _create_windows(df, final_op, final_cond, label_col, window_size, stride_train)

    # Split
    X_tv, X_te, Xc_tv, Xc_te, y_tv, y_te = train_test_split(
        X_op, X_cond, y, test_size=test_size, stratify=y, random_state=random_state)
    X_tr, X_va, Xc_tr, Xc_va, y_tr, y_va = train_test_split(
        X_tv, Xc_tv, y_tv, test_size=val_size, stratify=y_tv, random_state=random_state)

    train_ds = CSVFuelCellDataset(X_tr, Xc_tr, y_tr, augment=augment_train)
    val_ds = CSVFuelCellDataset(X_va, Xc_va, y_va)
    test_ds = CSVFuelCellDataset(X_te, Xc_te, y_te)

    meta = {
        "label_names": label_names,
        "num_classes": num_classes,
        "op_cols": final_op,
        "cond_cols": final_cond,
        "n_op_channels": len(final_op),
        "n_cond_dims": len(final_cond),
        "op_mean": op_mean,
        "op_std": op_std,
        "samples_per_class_train": np.bincount(y_tr, minlength=num_classes).tolist(),
        "window_size": window_size,
        "train_size": len(y_tr),
        "val_size": len(y_va),
        "test_size": len(y_te),
    }

    return train_ds, val_ds, test_ds, meta
