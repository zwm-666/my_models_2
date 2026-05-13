"""Feature engineering for PEMFC fault classification.

Transforms raw sensor readings into physics-informed features
for the CAPT-UniShape model.  All functions operate on a
``pandas.DataFrame`` and return it with new columns appended.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    op_cols: List[str],
    cond_cols: List[str],
    *,
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
    roll_window: int = 5,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Apply full feature-engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CSV data, already sorted by time.
    op_cols : list of str
        Raw operational sensor column names.
    cond_cols : list of str
        Condition set-point column names.
    clip_quantiles : tuple of float
        Lower / upper quantile for outlier clipping.
    roll_window : int
        Rolling-statistics window size.

    Returns
    -------
    df : pd.DataFrame
        With new columns appended.
    final_op_cols : list of str
        Final operational channel names (raw + engineered).
    cond_cols : list of str
        Unchanged condition column names.
    """
    df = df.copy()

    # 1. Outlier clipping on raw sensors
    _clip_outliers(df, op_cols, clip_quantiles)

    new_cols: List[str] = []

    # 2. Auto-detect and build physics features based on available columns
    cols = set(df.columns)

    # --- Voltage / Current derived ---
    if "U_totV" in cols and "iA" in cols:
        df["v_i_ratio"] = _safe_div(df["U_totV"], df["iA"])
        new_cols.append("v_i_ratio")
    if "PW" in cols and "iA" in cols and "U_totV" in cols:
        df["power_efficiency"] = _safe_div(df["PW"], df["iA"] * df["U_totV"])
        new_cols.append("power_efficiency")

    # --- Stoichiometry ---
    if "m_Air" in cols and "m_Air_write" in cols:
        df["air_stoich"] = _safe_div(df["m_Air"], df["m_Air_write"], fill=1.0)
        new_cols.append("air_stoich")
    if "m_H2" in cols and "m_H2_write" in cols:
        df["h2_stoich"] = _safe_div(df["m_H2"], df["m_H2_write"], fill=1.0)
        new_cols.append("h2_stoich")

    # --- Temperature ---
    temp_probes = [c for c in ["T_1", "T_2", "T_3", "T_4"] if c in cols]
    if temp_probes:
        df["T_mean"] = df[temp_probes].mean(axis=1)
        df["T_std"] = df[temp_probes].std(axis=1).fillna(0)
        df["T_max_min"] = df[temp_probes].max(axis=1) - df[temp_probes].min(axis=1)
        new_cols += ["T_mean", "T_std", "T_max_min"]
        if "T_Stack_inlet" in cols:
            df["T_stack_diff"] = df["T_Stack_inlet"] - df["T_mean"]
            new_cols.append("T_stack_diff")

    # --- Pressure ---
    if "P_Air_supply" in cols and "P_Air_inlet" in cols:
        df["P_air_drop"] = df["P_Air_supply"] - df["P_Air_inlet"]
        new_cols.append("P_air_drop")
    if "P_H2_supply" in cols and "P_H2_inlet" in cols:
        df["P_h2_drop"] = df["P_H2_supply"] - df["P_H2_inlet"]
        new_cols.append("P_h2_drop")
    if "P_Air_inlet" in cols and "P_H2_inlet" in cols:
        df["P_ratio"] = _safe_div(df["P_Air_inlet"], df["P_H2_inlet"], fill=1.0)
        new_cols.append("P_ratio")

    # --- Humidity ---
    if "RH_Air" in cols and "RH_H2" in cols:
        df["RH_diff"] = df["RH_Air"] - df["RH_H2"]
        df["RH_mean"] = (df["RH_Air"] + df["RH_H2"]) / 2
        new_cols += ["RH_diff", "RH_mean"]

    # --- First-order differences ---
    diff_candidates = ["U_totV", "iA", "PW", "T_mean", "P_air_drop", "RH_mean"]
    for col in diff_candidates:
        if col in df.columns:
            df[f"d_{col}"] = df[col].diff().fillna(0)
            new_cols.append(f"d_{col}")

    # --- Rolling statistics ---
    roll_candidates = ["U_totV", "iA", "T_mean", "P_air_drop"]
    for col in roll_candidates:
        if col in df.columns:
            r = df[col].rolling(roll_window, min_periods=1, center=True)
            df[f"{col}_rmean"] = r.mean()
            df[f"{col}_rstd"] = r.std().fillna(0)
            new_cols += [f"{col}_rmean", f"{col}_rstd"]

    # Assemble final list
    final_op_cols = list(op_cols) + new_cols

    # NaN cleanup
    df[final_op_cols] = df[final_op_cols].fillna(0)
    df[cond_cols] = df[cond_cols].fillna(0)

    return df, final_op_cols, cond_cols


def normalize_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Z-score normalise *columns* in place and return statistics."""
    df = df.copy()
    means = df[columns].mean().to_dict()
    stds = df[columns].std().replace(0, 1).to_dict()
    for c in columns:
        df[c] = (df[c] - means[c]) / stds[c]
    return df, means, stds


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(fill).clip(-10, 10)


def _clip_outliers(
    df: pd.DataFrame,
    cols: List[str],
    quantiles: Tuple[float, float],
) -> None:
    lo, hi = quantiles
    for col in cols:
        if col in df.columns:
            q_lo = df[col].quantile(lo)
            q_hi = df[col].quantile(hi)
            df[col] = df[col].clip(q_lo, q_hi)
