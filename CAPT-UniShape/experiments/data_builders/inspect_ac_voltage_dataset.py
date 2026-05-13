"""Inspect the AC voltage response dataset for PEMFC diagnosis.

The downloaded dataset contains six processed CSV matrices and the underlying
Gamry-style DTA files.  This script keeps the first pass deliberately light:
it streams the large CSV files, records labels/domains, checks curve length and
numeric quality, and writes paper-friendly notes about how to use the dataset
for diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LABEL_MAP = {
    "normal": 0,
    "drying": 1,
    "starvation": 2,
}


@dataclass(frozen=True)
class AcVoltageMetadata:
    domain: str
    label_name: str
    label_id: int


def infer_ac_voltage_metadata(path: Path) -> AcVoltageMetadata:
    """Infer MEA domain and fault label from a processed CSV or DTA path."""
    lower_parts = [part.lower() for part in path.parts]
    lower_text = "/".join(lower_parts)
    filename = path.name.lower()

    if "new_mea" in lower_text or "new mea" in lower_text:
        domain = "new_mea"
    elif "old_mea" in lower_text or "original mea" in lower_text:
        domain = "old_mea"
    elif filename.startswith(("drying_", "normal_", "starvation_")):
        domain = "old_mea"
    else:
        domain = "unknown"

    # DTA files often start with "Normal_PWR..." even when the folder label is
    # drying or starvation, so abnormal labels must take precedence.
    for label_name in ("starvation", "drying", "normal"):
        if label_name in lower_text:
            return AcVoltageMetadata(
                domain=domain,
                label_name=label_name,
                label_id=LABEL_MAP[label_name],
            )

    raise ValueError(f"Unable to infer AC voltage label from path: {path}")


def _safe_float(value: float | None, digits: int = 8) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def summarize_processed_csv(csv_path: Path, sample_rows: int = 20) -> dict[str, Any]:
    """Stream a processed response matrix and collect shape/quality statistics."""
    row_count = 0
    column_count: int | None = None
    inconsistent_rows = 0
    sample_arrays: list[np.ndarray[Any, Any]] = []
    trailing_values: set[int] = set()
    trailing_column_candidate = True

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row_count += 1

            values = np.fromstring(stripped, sep=",", dtype=np.float64)
            if column_count is None:
                column_count = int(values.size)
            elif int(values.size) != column_count:
                inconsistent_rows += 1

            if values.size:
                tail = float(values[-1])
                if math.isfinite(tail) and tail.is_integer() and int(tail) in set(LABEL_MAP.values()):
                    trailing_values.add(int(tail))
                else:
                    trailing_column_candidate = False

            if len(sample_arrays) >= int(sample_rows):
                continue
            sample_arrays.append(values)

    has_trailing_label = bool(
        trailing_column_candidate
        and trailing_values
        and column_count is not None
        and column_count > 1
    )
    curve_column_count = int(column_count or 0) - (1 if has_trailing_label else 0)

    finite_values = 0
    total_values = 0
    value_min: float | None = None
    value_max: float | None = None
    sum_values = 0.0
    sum_sq_values = 0.0
    for values in sample_arrays:
        curve_values = values[:-1] if has_trailing_label else values
        total_values += int(curve_values.size)
        finite = curve_values[np.isfinite(curve_values)]
        finite_values += int(finite.size)
        if finite.size == 0:
            continue
        current_min = float(np.min(finite))
        current_max = float(np.max(finite))
        value_min = current_min if value_min is None else min(value_min, current_min)
        value_max = current_max if value_max is None else max(value_max, current_max)
        sum_values += float(np.sum(finite))
        sum_sq_values += float(np.sum(finite * finite))

    mean = sum_values / finite_values if finite_values else None
    if finite_values and mean is not None:
        variance = max(0.0, (sum_sq_values / finite_values) - mean * mean)
        std = math.sqrt(variance)
    else:
        std = None

    finite_ratio = finite_values / total_values if total_values else None
    return {
        "file": str(csv_path),
        "row_count": int(row_count),
        "column_count": int(column_count or 0),
        "curve_column_count": int(curve_column_count),
        "has_trailing_label_column": bool(has_trailing_label),
        "trailing_label_values": sorted(trailing_values) if has_trailing_label else [],
        "inconsistent_rows": int(inconsistent_rows),
        "sampled_rows": int(len(sample_arrays)),
        "finite_ratio": _safe_float(finite_ratio),
        "value_min": _safe_float(value_min),
        "value_max": _safe_float(value_max),
        "value_mean": _safe_float(mean),
        "value_std": _safe_float(std),
    }


def summarize_processed_dir(processed_dir: Path, sample_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(processed_dir.glob("*.csv")):
        meta = infer_ac_voltage_metadata(csv_path)
        summary = summarize_processed_csv(csv_path, sample_rows=sample_rows)
        source_label_values = summary.get("trailing_label_values", [])
        source_label_id = source_label_values[0] if len(source_label_values) == 1 else None
        rows.append(
            {
                "file_name": csv_path.name,
                "domain": meta.domain,
                "label_name": meta.label_name,
                "label_id": meta.label_id,
                "source_label_id": source_label_id,
                "needs_label_remap": source_label_id is not None and int(source_label_id) != int(meta.label_id),
                **summary,
            }
        )
    return rows


def summarize_dta_files(root: Path) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, int], int] = {}
    for dta_path in root.rglob("*.DTA"):
        meta = infer_ac_voltage_metadata(dta_path)
        key = (meta.domain, meta.label_name, meta.label_id)
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "domain": domain,
            "label_name": label_name,
            "label_id": label_id,
            "dta_file_count": count,
        }
        for (domain, label_name, label_id), count in sorted(counts.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_count_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| MEA域 | 类别 | label_id | 数量 |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {row['label_name']} | {row['label_id']} | {row['dta_file_count']} |"
        )
    return "\n".join(lines)


def _format_processed_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 文件 | MEA域 | 类别 | 样本行数 | 原始列数 | 曲线列数 | CSV标签 | 需重映射 | 有效值比例 | 值域 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        value_range = f"{row['value_min']} ~ {row['value_max']}"
        lines.append(
            "| "
            f"{row['file_name']} | {row['domain']} | {row['label_name']} | "
            f"{row['row_count']} | {row['column_count']} | {row['curve_column_count']} | "
            f"{row['source_label_id']} | {row['needs_label_remap']} | "
            f"{row['finite_ratio']} | {value_range} |"
        )
    return "\n".join(lines)


def write_diagnosis_markdown(
    path: Path,
    processed_rows: list[dict[str, Any]],
    dta_rows: list[dict[str, Any]],
    sample_rows: int,
) -> None:
    total_csv_samples = sum(int(row["row_count"]) for row in processed_rows)
    total_dta_files = sum(int(row["dta_file_count"]) for row in dta_rows)
    curve_lengths = sorted({int(row["curve_column_count"]) for row in processed_rows})
    text = f"""# AC Voltage Response 数据集诊断建议

## 数据体检结论

- 原始 `.DTA` 文件数：{total_dta_files}
- 已处理 CSV 样本行数：{total_csv_samples}
- CSV 曲线长度：{curve_lengths}
- 统计方式：每个 CSV 流式读取全量行数，并抽样前 {sample_rows} 行计算数值范围、均值、标准差和有效值比例。
- 注意：processed CSV 的最后一列是数据集自带标签列，不应作为电压曲线输入；生成 NPZ 时应剔除该列。
- 注意：processed CSV 自带标签编码与本文建议编码不完全一致，后续应按路径语义重映射为 `0=normal`、`1=drying`、`2=starvation`。

## DTA 类别与域分布

{_format_count_table(dta_rows)}

## 已处理 CSV 概览

{_format_processed_table(processed_rows)}

## 诊断任务设计

建议把该数据集作为独立的 AC 电压响应三分类诊断任务，而不是直接替换原 `测试数据.xlsx`。标签定义为 `0=normal`、`1=drying`、`2=starvation`。每条响应曲线可作为 UniShape 的形状输入，初始版本可构造 `x_op=[N,1,T]`；随后可从曲线中提取均值、标准差、峰谷值、稳态值、前后段斜率、响应面积和频域能量等统计量作为 `x_cond`。

最有论文价值的评估协议是跨 MEA 泛化：`old_mea -> new_mea` 和 `new_mea -> old_mea`。这比随机划分更能说明模型能否适应不同 MEA 的响应差异。常规混合分层划分可以作为上限对照，但不宜作为唯一主结果。

## 下一步建议

1. 先基于 processed CSV 生成 AC 专用 NPZ：`x_op=[N,1,T]`，`x_cond=[N,D]`，`labels=[N]`，并保留 `domain` 字段。
2. 训练三类基线：SVM、1D-CNN、CAPT-UniShape 适配版。
3. 重点报告 Accuracy、Macro-F1、Normal Recall、Drying/Starvation 混淆矩阵。
4. 完成跨 MEA 实验后，再考虑从 `.DTA` 直接解析 `Vf/Im/Sig/Temp` 多通道输入。
"""
    path.write_text(text, encoding="utf-8")


def inspect_dataset(data_root: Path, output_dir: Path, sample_rows: int) -> dict[str, Any]:
    processed_dir = data_root / "AC Voltage Responses" / "AC Voltage Responses" / "Processed_Data"
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed_Data directory not found: {processed_dir}")

    processed_rows = summarize_processed_dir(processed_dir, sample_rows=sample_rows)
    dta_rows = summarize_dta_files(data_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "processed_csv_summary.csv", processed_rows)
    _write_csv(output_dir / "dta_count_summary.csv", dta_rows)

    payload = {
        "data_root": str(data_root),
        "processed_dir": str(processed_dir),
        "label_map": LABEL_MAP,
        "processed_csv": processed_rows,
        "dta_counts": dta_rows,
    }
    (output_dir / "ac_voltage_dataset_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_diagnosis_markdown(
        output_dir / "AC电压响应数据集诊断建议.md",
        processed_rows=processed_rows,
        dta_rows=dta_rows,
        sample_rows=sample_rows,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/AC Voltage Response Data"),
        help="Root directory of the AC voltage response dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ac_voltage_dataset_diagnosis"),
        help="Directory for JSON/CSV/Markdown inspection outputs.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=20,
        help="Rows per processed CSV used for numeric range/statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = inspect_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        sample_rows=max(1, int(args.sample_rows)),
    )
    print(
        json.dumps(
            {
                "processed_csv_files": len(payload["processed_csv"]),
                "dta_groups": len(payload["dta_counts"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
