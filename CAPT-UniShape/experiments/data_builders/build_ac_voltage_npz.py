"""Build an NPZ dataset for AC voltage response diagnosis.

The processed CSV files contain one sample per row.  The final column is the
dataset's original label and must not be treated as a voltage point.  This
builder remaps labels by file/path semantics to the project convention:
0=normal, 1=drying, 2=starvation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inspect_ac_voltage_dataset import infer_ac_voltage_metadata


DOMAIN_MAP = {"old_mea": 0, "new_mea": 1}


def resample_curve(curve: np.ndarray[Any, Any], target_len: int) -> np.ndarray[Any, Any]:
    """Linearly resample a 1D curve to a fixed length."""
    values = np.asarray(curve, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot resample an empty curve")
    if int(target_len) <= 0:
        raise ValueError("target_len must be positive")
    if values.size == int(target_len):
        return values.astype(np.float32, copy=True)
    source_grid = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    target_grid = np.linspace(0.0, 1.0, int(target_len), dtype=np.float32)
    return np.interp(target_grid, source_grid, values).astype(np.float32)


def build_condition_features(curves: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return 12 compact statistical features per voltage response curve."""
    x = np.asarray(curves, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("curves must be a 2D array [N, T]")

    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    min_v = np.min(x, axis=1)
    max_v = np.max(x, axis=1)
    p05 = np.percentile(x, 5, axis=1)
    p50 = np.percentile(x, 50, axis=1)
    p95 = np.percentile(x, 95, axis=1)
    initial = x[:, 0]
    final = x[:, -1]
    delta = final - initial
    area = np.trapz(x, axis=1) / max(1, x.shape[1] - 1)
    peak_to_peak = max_v - min_v
    features = np.stack(
        [mean, std, min_v, max_v, p05, p50, p95, initial, final, delta, area, peak_to_peak],
        axis=1,
    )
    return features.astype(np.float32)


def build_eis_like_sequence(curves: np.ndarray[Any, Any], seq_len: int) -> np.ndarray[Any, Any]:
    """Create a four-channel shape sequence from voltage response curves."""
    x = np.asarray(curves, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("curves must be a 2D array [N, T]")
    resampled = np.stack([resample_curve(row, int(seq_len)) for row in x], axis=0)
    diff = np.gradient(resampled, axis=1).astype(np.float32)
    centered = resampled - np.mean(resampled, axis=1, keepdims=True)
    cumulative = np.cumsum(centered, axis=1).astype(np.float32)
    denom = np.max(np.abs(cumulative), axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    cumulative = cumulative / denom
    freq_axis = np.tile(np.linspace(0.0, 1.0, int(seq_len), dtype=np.float32), (x.shape[0], 1))
    return np.stack([resampled, diff, cumulative, freq_axis], axis=1).astype(np.float32)


def build_split_array(
    labels: np.ndarray[Any, Any],
    domains: np.ndarray[Any, Any],
    protocol: str,
    val_fraction: float,
    seed: int,
    test_fraction: float = 0.2,
) -> np.ndarray[Any, Any]:
    """Build split values: 0=train, 1=val, 2=test."""
    labels = np.asarray(labels, dtype=np.int64)
    domains = np.asarray(domains, dtype=np.int64)
    split = np.full(labels.shape, -1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    if protocol == "old_to_new":
        train_pool = np.where(domains == DOMAIN_MAP["old_mea"])[0]
        split[np.where(domains == DOMAIN_MAP["new_mea"])[0]] = 2
    elif protocol == "new_to_old":
        train_pool = np.where(domains == DOMAIN_MAP["new_mea"])[0]
        split[np.where(domains == DOMAIN_MAP["old_mea"])[0]] = 2
    elif protocol == "mixed_stratified":
        train_pool_parts: list[np.ndarray[Any, Any]] = []
        for label in sorted(np.unique(labels).tolist()):
            idx = np.where(labels == int(label))[0]
            idx = rng.permutation(idx)
            n_test = max(1, int(round(idx.size * float(test_fraction))))
            split[idx[:n_test]] = 2
            train_pool_parts.append(idx[n_test:])
        train_pool = np.concatenate(train_pool_parts)
    else:
        raise ValueError("protocol must be old_to_new, new_to_old or mixed_stratified")

    for label in sorted(np.unique(labels[train_pool]).tolist()):
        idx = train_pool[labels[train_pool] == int(label)]
        idx = rng.permutation(idx)
        n_val = max(1, int(round(idx.size * float(val_fraction))))
        split[idx[:n_val]] = 1
        split[idx[n_val:]] = 0

    if np.any(split < 0):
        raise ValueError("Some samples were not assigned to a split")
    return split


def _standardize_feature_matrix(
    x: np.ndarray[Any, Any],
    train_mask: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    mean = np.mean(x[train_mask], axis=0, keepdims=True)
    std = np.std(x[train_mask], axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return ((x - mean) / std).astype(np.float32), {
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
    }


def _standardize_sequence_channels(
    x: np.ndarray[Any, Any],
    train_mask: np.ndarray[Any, Any],
    channels: list[int],
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    out = x.astype(np.float32, copy=True)
    stats: dict[str, Any] = {}
    for channel in channels:
        values = out[train_mask, channel, :]
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not math.isfinite(std) or std <= 1e-6:
            std = 1.0
        out[:, channel, :] = (out[:, channel, :] - mean) / std
        stats[str(channel)] = {"mean": mean, "std": std}
    return out, stats


def load_processed_curves(
    processed_dir: Path,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], list[dict[str, Any]]]:
    curves: list[np.ndarray[Any, Any]] = []
    labels: list[int] = []
    domains: list[int] = []
    source_labels: list[int] = []
    manifest: list[dict[str, Any]] = []

    for csv_path in sorted(processed_dir.glob("*.csv")):
        meta = infer_ac_voltage_metadata(csv_path)
        domain_id = DOMAIN_MAP[meta.domain]
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row_index, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue
                values = np.fromstring(stripped, sep=",", dtype=np.float32)
                if values.size < 2:
                    raise ValueError(f"Expected at least two columns in {csv_path}")
                curve = values[:-1].astype(np.float32)
                source_label = int(round(float(values[-1])))
                curves.append(curve)
                labels.append(int(meta.label_id))
                domains.append(int(domain_id))
                source_labels.append(source_label)
                manifest.append(
                    {
                        "file_name": csv_path.name,
                        "row_index": row_index,
                        "domain": meta.domain,
                        "domain_id": domain_id,
                        "label_name": meta.label_name,
                        "label_id": int(meta.label_id),
                        "source_label_id": source_label,
                    }
                )

    if not curves:
        raise ValueError(f"No processed CSV rows found in {processed_dir}")
    lengths = {curve.size for curve in curves}
    if len(lengths) != 1:
        raise ValueError(f"Processed curves have inconsistent lengths: {sorted(lengths)}")
    return (
        np.stack(curves, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(domains, dtype=np.int64),
        np.asarray(source_labels, dtype=np.int64),
        manifest,
    )


def select_domain_subset(
    curves: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    domains: np.ndarray[Any, Any],
    source_labels: np.ndarray[Any, Any],
    manifest: list[dict[str, Any]],
    domain_filter: str | None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], list[dict[str, Any]]]:
    if not domain_filter:
        return curves, labels, domains, source_labels, manifest
    if domain_filter not in DOMAIN_MAP:
        raise ValueError(f"domain_filter must be one of {sorted(DOMAIN_MAP)}")

    domain_id = DOMAIN_MAP[domain_filter]
    keep_mask = np.asarray(domains, dtype=np.int64) == int(domain_id)
    if not np.any(keep_mask):
        raise ValueError(f"No samples found for domain_filter={domain_filter}")
    kept_manifest = [item for item, keep in zip(manifest, keep_mask.tolist()) if keep]
    return (
        curves[keep_mask],
        labels[keep_mask],
        domains[keep_mask],
        source_labels[keep_mask],
        kept_manifest,
    )


def _count_by(values: np.ndarray[Any, Any]) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def resolve_processed_dir(data_root: Path) -> Path:
    candidates = [
        data_root / "AC Voltage Responses" / "Processed_Data",
        data_root / "AC Voltage Responses" / "AC Voltage Responses" / "Processed_Data",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ValueError(f"Could not find Processed_Data under {data_root}")


def build_ac_voltage_npz(
    data_root: Path,
    output: Path,
    protocol: str,
    op_seq_len: int,
    eis_seq_len: int,
    val_fraction: float,
    seed: int,
    test_fraction: float = 0.2,
    domain_filter: str | None = None,
) -> dict[str, Any]:
    processed_dir = resolve_processed_dir(data_root)
    curves, labels, domains, source_labels, manifest = load_processed_curves(processed_dir)
    curves, labels, domains, source_labels, manifest = select_domain_subset(
        curves,
        labels,
        domains,
        source_labels,
        manifest,
        domain_filter=domain_filter,
    )
    split = build_split_array(
        labels,
        domains,
        protocol=protocol,
        val_fraction=val_fraction,
        seed=seed,
        test_fraction=test_fraction,
    )
    train_mask = split == 0

    op_curves = np.stack([resample_curve(row, int(op_seq_len)) for row in curves], axis=0)
    x_op = op_curves[:, None, :].astype(np.float32)
    x_eis = build_eis_like_sequence(curves, seq_len=int(eis_seq_len))
    x_cond = build_condition_features(curves)

    x_op, op_norm = _standardize_sequence_channels(x_op, train_mask, channels=[0])
    x_eis, eis_norm = _standardize_sequence_channels(x_eis, train_mask, channels=[0, 1, 2])
    x_cond, cond_norm = _standardize_feature_matrix(x_cond, train_mask)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x_op=x_op.astype(np.float32),
        x_eis=x_eis.astype(np.float32),
        x_cond=x_cond.astype(np.float32),
        labels=labels.astype(np.int64),
        y=labels.astype(np.int64),
        split=split.astype(np.int64),
        domain_id=domains.astype(np.int64),
        source_label_id=source_labels.astype(np.int64),
    )

    split_counts = {
        name: _count_by(labels[split == value])
        for name, value in {"train": 0, "val": 1, "test": 2}.items()
    }
    summary = {
        "data_root": str(data_root),
        "processed_dir": str(processed_dir),
        "output": str(output),
        "protocol": protocol,
        "domain_filter": domain_filter,
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "label_map": {"normal": 0, "drying": 1, "starvation": 2},
        "domain_map": DOMAIN_MAP,
        "source_label_note": "Final CSV column is removed from x_op/x_eis and retained as source_label_id only.",
        "shape": {
            "curves_raw": list(curves.shape),
            "x_op": list(x_op.shape),
            "x_eis": list(x_eis.shape),
            "x_cond": list(x_cond.shape),
            "labels": list(labels.shape),
        },
        "split_counts": split_counts,
        "domain_counts": _count_by(domains),
        "label_counts": _count_by(labels),
        "normalization": {"x_op": op_norm, "x_eis": eis_norm, "x_cond": cond_norm},
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = output.with_suffix(".manifest.csv")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(manifest[0].keys()) + ["split"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item, split_value in zip(manifest, split):
            row = dict(item)
            row["split"] = int(split_value)
            writer.writerow(row)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/AC Voltage Response Data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ac_voltage_response_old_to_new_w256.npz"))
    parser.add_argument("--protocol", choices=["old_to_new", "new_to_old", "mixed_stratified"], default="old_to_new")
    parser.add_argument("--op-seq-len", type=int, default=256)
    parser.add_argument("--eis-seq-len", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--domain-filter", choices=sorted(DOMAIN_MAP), default=None)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_ac_voltage_npz(
        data_root=args.data_root,
        output=args.output,
        protocol=args.protocol,
        op_seq_len=args.op_seq_len,
        eis_seq_len=args.eis_seq_len,
        val_fraction=args.val_fraction,
        seed=args.seed,
        test_fraction=args.test_fraction,
        domain_filter=args.domain_filter,
    )
    print(json.dumps({"output": summary["output"], "shape": summary["shape"], "split_counts": summary["split_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
