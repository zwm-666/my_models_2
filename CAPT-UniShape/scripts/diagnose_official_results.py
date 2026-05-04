"""Diagnose CAPT-UniShape result artifacts for split and per-class failure modes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_error": f"invalid_json: {exc}"}
    if isinstance(payload, dict):
        return payload
    return {"_error": "json_root_is_not_object"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _report_average_score(metrics: dict[str, Any], average_key: str, score_key: str) -> float:
    report = metrics.get("classification_report")
    if not isinstance(report, dict):
        return 0.0
    average = report.get(average_key)
    if not isinstance(average, dict):
        return 0.0
    return _safe_float(average.get(score_key))


def _metric_score(metrics: dict[str, Any], metric_key: str) -> float:
    direct = metrics.get(metric_key)
    if direct is not None:
        return _safe_float(direct)
    if metric_key == "macro_f1":
        return _report_average_score(metrics, "macro avg", "f1-score")
    if metric_key == "weighted_f1":
        return _report_average_score(metrics, "weighted avg", "f1-score")
    return 0.0


def _infer_ratio_model(metrics_path: Path, results_root: Path) -> tuple[str, str]:
    try:
        relative = metrics_path.relative_to(results_root)
    except ValueError:
        relative = metrics_path
    parts = relative.parts
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    if len(parts) >= 2:
        return "", parts[-2]
    return "", ""


def _candidate_summaries(data_root: Path, ratio: str) -> Iterable[Path]:
    if ratio:
        yield data_root / f"official_self_stack_impedance_eis_w64_{ratio}.summary.json"
        yield data_root / f"official_self_stack_impedance_eis_w64_{ratio}_fixed_base.summary.json"
    yield from sorted(data_root.rglob("*.summary.json"))


def _load_matching_summary(data_root: Path, ratio: str) -> tuple[Path | None, dict[str, Any]]:
    seen: set[Path] = set()
    for candidate in _candidate_summaries(data_root, ratio):
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        if ratio and ratio not in candidate.name and "fixed_base" not in candidate.name:
            continue
        summary = _load_json(candidate)
        if summary:
            return candidate, summary
    return None, {}


def _load_fixed_base_summary(data_root: Path) -> tuple[Path | None, dict[str, Any]]:
    for candidate in sorted(data_root.rglob("*fixed_base.summary.json")):
        summary = _load_json(candidate)
        if _split_quality(summary):
            return candidate, summary
    return None, {}


def _split_quality(summary: dict[str, Any]) -> dict[str, Any]:
    quality = summary.get("split_quality")
    if isinstance(quality, dict):
        return quality
    source_meta = summary.get("source_meta")
    if isinstance(source_meta, dict):
        nested = source_meta.get("split_quality")
        if isinstance(nested, dict):
            return nested
    return {}


def _split_counts(summary: dict[str, Any], split_name: str) -> dict[str, int]:
    split_counts = summary.get("split_label_counts")
    if not isinstance(split_counts, dict):
        return {}
    counts = split_counts.get(split_name)
    if not isinstance(counts, dict):
        return {}
    return {str(label): int(count) for label, count in counts.items()}


def _per_class_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = metrics.get("per_class_f1")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    report = metrics.get("classification_report")
    if not isinstance(report, dict):
        return []
    parsed: list[dict[str, Any]] = []
    for key, value in report.items():
        if not str(key).isdigit() or not isinstance(value, dict):
            continue
        parsed.append(
            {
                "class_id": int(key),
                "precision": _safe_float(value.get("precision")),
                "recall": _safe_float(value.get("recall")),
                "f1": _safe_float(value.get("f1-score")),
                "support": int(_safe_float(value.get("support"))),
            }
        )
    return parsed


def _worst_class(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = _per_class_rows(metrics)
    if not rows:
        return {"class_id": "", "recall": "", "f1": "", "support": ""}
    return min(rows, key=lambda row: (_safe_float(row.get("recall")), _safe_float(row.get("f1"))))


def _min_support(metrics: dict[str, Any]) -> int:
    supports = [int(_safe_float(row.get("support"))) for row in _per_class_rows(metrics)]
    return min(supports) if supports else 0


def diagnose_one(
    metrics_path: Path,
    results_root: Path,
    data_root: Path,
    test_macro_threshold: float,
    gap_threshold: float,
    recall_threshold: float,
    min_support_threshold: int,
) -> dict[str, Any]:
    payload = _load_json(metrics_path)
    ratio, model = _infer_ratio_model(metrics_path, results_root)
    test_payload = payload.get("test")
    test_metrics = test_payload if isinstance(test_payload, dict) else {}
    val_metrics = payload
    summary_path, summary = _load_matching_summary(data_root, ratio)
    quality = _split_quality(summary)
    quality_source = str(summary_path) if summary_path and quality else ""
    if not quality:
        fixed_base_path, fixed_base_summary = _load_fixed_base_summary(data_root)
        fixed_base_quality = _split_quality(fixed_base_summary)
        if fixed_base_quality:
            quality = fixed_base_quality
            quality_source = str(fixed_base_path) if fixed_base_path else ""
    worst = _worst_class(test_metrics)
    val_macro = _metric_score(val_metrics, "macro_f1")
    test_macro = _metric_score(test_metrics, "macro_f1")
    gap = val_macro - test_macro
    min_test_support = _min_support(test_metrics)
    if min_test_support == 0:
        test_counts = _split_counts(summary, "test")
        min_test_support = min(test_counts.values()) if test_counts else 0
    issues: list[str] = []
    if not test_metrics:
        issues.append("missing_test_metrics")
    if test_metrics and test_macro < test_macro_threshold:
        issues.append(f"low_test_macro_f1<{test_macro_threshold:.2f}")
    if test_metrics and gap > gap_threshold:
        issues.append(f"large_val_test_gap>{gap_threshold:.2f}")
    worst_recall = _safe_float(worst.get("recall"), default=1.0)
    if test_metrics and worst_recall < recall_threshold:
        issues.append(f"low_worst_class_recall<{recall_threshold:.2f}")
    if min_test_support and min_test_support < min_support_threshold:
        issues.append(f"small_test_class_support<{min_support_threshold}")
    if quality and quality.get("passed") is False:
        issues.append("split_quality_failed")
    if payload.get("refit_trainval") is True:
        issues.append("refit_trainval_check_selection_vs_in_sample")
    return {
        "ratio": ratio,
        "model": model,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path) if summary_path else "",
        "val_macro_f1": val_macro,
        "test_macro_f1": test_macro if test_metrics else "",
        "val_test_macro_gap": gap if test_metrics else "",
        "test_accuracy": _safe_float(test_metrics.get("accuracy")) if test_metrics else "",
        "test_weighted_f1": _metric_score(test_metrics, "weighted_f1") if test_metrics else "",
        "worst_class_id": worst.get("class_id", ""),
        "worst_class_recall": worst.get("recall", ""),
        "worst_class_f1": worst.get("f1", ""),
        "min_test_class_support": min_test_support,
        "split_quality_passed": quality.get("passed", ""),
        "split_quality_source": quality_source,
        "min_val_class_windows": quality.get("min_val_class_windows", ""),
        "min_test_class_windows": quality.get("min_test_class_windows", ""),
        "min_val_class_groups": quality.get("min_val_class_groups", ""),
        "min_test_class_groups": quality.get("min_test_class_groups", ""),
        "refit_trainval": payload.get("refit_trainval", ""),
        "model_selection_rule": payload.get("model_selection_rule", ""),
        "issues": ";".join(issues),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ratio",
        "model",
        "val_macro_f1",
        "test_macro_f1",
        "val_test_macro_gap",
        "test_accuracy",
        "test_weighted_f1",
        "worst_class_id",
        "worst_class_recall",
        "worst_class_f1",
        "min_test_class_support",
        "split_quality_passed",
        "split_quality_source",
        "min_val_class_windows",
        "min_test_class_windows",
        "min_val_class_groups",
        "min_test_class_groups",
        "refit_trainval",
        "model_selection_rule",
        "issues",
        "metrics_path",
        "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 CAPT-UniShape 结果产物中的 split 与 per-class 失败模式")
    parser.add_argument("--results-root", default="results/official_baseline_comparison")
    parser.add_argument("--data-root", default="data/processed/official_baseline_comparison")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-macro-threshold", type=float, default=0.85)
    parser.add_argument("--gap-threshold", type=float, default=0.15)
    parser.add_argument("--recall-threshold", type=float, default=0.70)
    parser.add_argument("--min-support-threshold", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = _project_path(args.results_root)
    data_root = _project_path(args.data_root)
    output_dir = _project_path(args.output_dir) if args.output_dir else results_root / "diagnosis"
    metric_paths = sorted(results_root.rglob("metrics.json"))
    rows = [
        diagnose_one(
            metrics_path,
            results_root,
            data_root,
            test_macro_threshold=float(args.test_macro_threshold),
            gap_threshold=float(args.gap_threshold),
            recall_threshold=float(args.recall_threshold),
            min_support_threshold=int(args.min_support_threshold),
        )
        for metrics_path in metric_paths
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "diagnosis_summary.csv", rows)
    (output_dir / "diagnosis_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"扫描 metrics.json 数量: {len(rows)}", flush=True)
    print(f"诊断 CSV: {output_dir / 'diagnosis_summary.csv'}", flush=True)
    print(f"诊断 JSON: {output_dir / 'diagnosis_summary.json'}", flush=True)
    flagged = [row for row in rows if row.get("issues")]
    print(f"存在诊断标记的运行: {len(flagged)}", flush=True)
    for row in flagged[:10]:
        print(
            " | ".join(
                [
                    f"ratio={row['ratio']}",
                    f"model={row['model']}",
                    f"test_macro_f1={row['test_macro_f1']}",
                    f"worst_class={row['worst_class_id']}",
                    f"issues={row['issues']}",
                ]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
