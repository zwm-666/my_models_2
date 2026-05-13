"""Run proposed-only experiments until the target test accuracy is reached."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = ROOT / "scripts" / "run_official_baseline_experiments.py"
RATIO_KEYS = ["8_2", "7_3", "6_4", "5_5"]
CLASS_WEIGHTING_KEYS = [
    "sqrt_balanced",
    "effective_number",
    "logit_adjusted",
    "balanced_softmax",
    "balanced",
    "none",
]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def metric_record_from_payload(path: Path, payload: dict[str, Any], source: str) -> dict[str, Any]:
    """Convert a metrics.json payload into one comparable search record."""
    test_payload = payload.get("test")
    if isinstance(test_payload, dict) and "accuracy" in test_payload:
        metrics = test_payload
        metric_source = "test"
    else:
        metrics = payload
        metric_source = "top_level"
    return {
        "metrics_path": str(path),
        "source": source,
        "metric_source": metric_source,
        "test_accuracy": _safe_float(metrics.get("accuracy")),
        "test_macro_f1": _safe_float(metrics.get("macro_f1")),
        "test_weighted_f1": _safe_float(metrics.get("weighted_f1")),
        "val_accuracy": _safe_float(payload.get("accuracy")),
        "val_macro_f1": _safe_float(payload.get("macro_f1")),
        "class_weighting": payload.get("class_weighting", ""),
        "model_selection_rule": payload.get("model_selection_rule", ""),
        "refit_trainval": bool(payload.get("refit_trainval", False)),
        "split_diagnostics": payload.get("split_diagnostics", {}),
    }


def load_metric_record(path: Path, source: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return metric_record_from_payload(path, payload, source=source)


def first_successful_record(
    records: Iterable[dict[str, Any]],
    threshold: float,
    metric_key: str = "val_macro_f1",
) -> dict[str, Any] | None:
    for record in records:
        if _safe_float(record.get(metric_key)) >= float(threshold):
            return record
    return None


def discover_metrics(paths: Iterable[str | Path]) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in paths:
        path = _project_path(raw_path)
        if path.is_file():
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(sorted(path.rglob("metrics.json")))
    return discovered


@dataclass(frozen=True)
class ProposedAttempt:
    ratio: str
    seed: int
    class_weighting: str
    lr: float
    weight_decay: float
    refit_trainval: bool

    @property
    def name(self) -> str:
        suffix = "_refit" if self.refit_trainval else ""
        return (
            f"seed_{self.seed}_{self.ratio}_{self.class_weighting}"
            f"_lr{_float_token(self.lr)}_wd{_float_token(self.weight_decay)}{suffix}"
        )


def build_attempts(args: argparse.Namespace) -> list[ProposedAttempt]:
    refit_values = [False, True] if bool(args.include_refit_trainval) else [bool(args.refit_trainval)]
    attempts = [
        ProposedAttempt(
            ratio=ratio,
            seed=seed,
            class_weighting=class_weighting,
            lr=lr,
            weight_decay=weight_decay,
            refit_trainval=refit,
        )
        for seed in args.seeds
        for ratio in args.ratios
        for class_weighting in args.class_weightings
        for lr in args.lrs
        for weight_decay in args.weight_decays
        for refit in refit_values
    ]
    if args.max_attempts is not None:
        return attempts[: max(0, int(args.max_attempts))]
    return attempts


def _append_optional(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend([name, str(value)])


def attempt_result_root(args: argparse.Namespace, attempt: ProposedAttempt) -> Path:
    return _project_path(args.output_root) / attempt.name


def attempt_data_root(args: argparse.Namespace, attempt: ProposedAttempt) -> Path:
    return _project_path(args.data_root) / attempt.name


def attempt_metrics_path(args: argparse.Namespace, attempt: ProposedAttempt) -> Path:
    return attempt_result_root(args, attempt) / attempt.ratio / "proposed" / "metrics.json"


def attempt_command(args: argparse.Namespace, attempt: ProposedAttempt) -> list[str]:
    command = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--excel",
        str(args.excel),
        "--ratios",
        attempt.ratio,
        "--models",
        "proposed",
        "--output-root",
        str(attempt_result_root(args, attempt)),
        "--data-root",
        str(attempt_data_root(args, attempt)),
        "--seed",
        str(attempt.seed),
        "--split-protocol",
        str(args.split_protocol),
        "--fixed-test-ratio",
        str(args.fixed_test_ratio),
        "--window-size",
        str(args.window_size),
        "--stride-train",
        str(args.stride_train),
        "--stride-eval",
        str(args.stride_eval),
        "--eis-seq-len",
        str(args.eis_seq_len),
        "--split-mode",
        str(args.split_mode),
        "--segment-gap-seconds",
        str(args.segment_gap_seconds),
        "--segment-block-seconds",
        str(args.segment_block_seconds),
        "--group-split-strategy",
        str(args.group_split_strategy),
        "--val-size",
        str(args.val_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--min-epochs-before-stop",
        str(args.min_epochs_before_stop),
        "--val-metric-smoothing",
        str(args.val_metric_smoothing),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(attempt.lr),
        "--weight-decay",
        str(attempt.weight_decay),
        "--class-weighting",
        attempt.class_weighting,
        "--proposed-config",
        str(args.proposed_config),
        "--checkpoint-selection",
        str(args.checkpoint_selection),
        "--split-retries",
        str(args.split_retries),
        "--min-eval-class-windows",
        str(args.min_eval_class_windows),
        "--min-eval-class-groups",
        str(args.min_eval_class_groups),
    ]
    command.append("--refit-trainval" if attempt.refit_trainval else "--no-refit-trainval")
    if not bool(args.segment_label_boundary):
        command.append("--no-segment-label-boundary")
    if bool(args.class_aware_train_stride):
        command.append("--class-aware-train-stride")
    _append_optional(command, "--min-train-stride", args.min_train_stride)
    _append_optional(command, "--max-train-stride", args.max_train_stride)
    return command


def _write_summary(
    output_root: Path,
    records: list[dict[str, Any]],
    winner: dict[str, Any] | None,
    threshold: float,
    search_metric: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_threshold": float(threshold),
        "search_metric": search_metric,
        "test_metrics_are_for_reporting_only": True,
        "winner": winner,
        "attempts": records,
    }
    (output_root / "search_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "source",
        "metric_source",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "val_accuracy",
        "val_macro_f1",
        "search_metric",
        "class_weighting",
        "model_selection_rule",
        "refit_trainval",
        "metrics_path",
        "command",
    ]
    with (output_root / "search_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            record = dict(record)
            record["search_metric"] = search_metric
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def _run_attempt(args: argparse.Namespace, attempt: ProposedAttempt) -> dict[str, Any]:
    metrics_path = attempt_metrics_path(args, attempt)
    command = attempt_command(args, attempt)
    if bool(args.reuse_completed) and metrics_path.exists():
        record = load_metric_record(metrics_path, source="reused")
        record["command"] = " ".join(command)
        return record
    result_root = attempt_result_root(args, attempt)
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "command.txt").write_text(" ".join(command), encoding="utf-8")
    subprocess.run(command, cwd=ROOT, check=True)
    record = load_metric_record(metrics_path, source="trained")
    record["command"] = " ".join(command)
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只跑 proposed 模型，直到测试准确率达到阈值")
    parser.add_argument("--excel", default="data/raw/水淹和膜干故障测试数据_补充特征汇总.xlsx")
    parser.add_argument("--output-root", default="results/codex_proposed_accuracy_search")
    parser.add_argument("--data-root", default="data/processed/codex_proposed_accuracy_search")
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--search-metric", choices=["val_macro_f1", "val_accuracy"], default="val_macro_f1", help="Metric used to stop hyperparameter search; test metrics are reported but not used for search by default")
    parser.add_argument("--candidate-metrics", nargs="*", default=[], help="先扫描已有 metrics.json 文件或目录，命中阈值则直接记录为 winner")
    parser.add_argument("--ignore-candidate-success", action="store_true", help="即使候选结果达标也继续重新训练")
    parser.add_argument("--reuse-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--ratios", nargs="+", default=["8_2"], choices=RATIO_KEYS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[44, 42, 43])
    parser.add_argument("--class-weightings", nargs="+", default=["sqrt_balanced", "effective_number", "logit_adjusted"], choices=CLASS_WEIGHTING_KEYS)
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3e-4])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--split-protocol", choices=["fixed_test", "independent"], default="fixed_test")
    parser.add_argument("--fixed-test-ratio", choices=RATIO_KEYS, default="8_2")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride-train", type=int, default=16)
    parser.add_argument("--stride-eval", type=int, default=32)
    parser.add_argument("--eis-seq-len", type=int, default=128)
    parser.add_argument("--split-mode", default="segment")
    parser.add_argument("--segment-gap-seconds", type=float, default=600.0)
    parser.add_argument("--segment-block-seconds", type=float, default=300.0)
    parser.add_argument("--segment-label-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group-split-strategy", choices=["holdout_first", "three_way", "two_stage"], default="holdout_first")
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--refit-trainval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-refit-trainval", action="store_true")
    parser.add_argument("--min-epochs-before-stop", type=int, default=20)
    parser.add_argument("--val-metric-smoothing", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--class-aware-train-stride", action="store_true")
    parser.add_argument("--min-train-stride", type=int, default=None)
    parser.add_argument("--max-train-stride", type=int, default=None)
    parser.add_argument("--proposed-config", default="configs/rbf_kanfusion.yaml")
    parser.add_argument("--checkpoint-selection", choices=["best_val", "best-val", "last"], default="best_val")
    parser.add_argument("--split-retries", type=int, default=50)
    parser.add_argument("--min-eval-class-windows", type=int, default=5)
    parser.add_argument("--min-eval-class-groups", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = _project_path(args.output_root)
    records: list[dict[str, Any]] = []
    if args.candidate_metrics:
        for metrics_path in discover_metrics(args.candidate_metrics):
            try:
                records.append(load_metric_record(metrics_path, source="candidate"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"跳过无法读取的候选指标 {metrics_path}: {exc}", flush=True)
        winner = first_successful_record(records, threshold=args.target_accuracy, metric_key=args.search_metric)
        if winner is not None and not bool(args.ignore_candidate_success):
            _write_summary(output_root, records, winner, args.target_accuracy, args.search_metric)
            print(
                f"候选结果的搜索指标已达标: {args.search_metric}={winner[args.search_metric]:.4f}, "
                f"test_acc={winner['test_accuracy']:.4f}, "
                f"metrics={winner['metrics_path']}",
                flush=True,
            )
            return
    attempts = build_attempts(args)
    if args.dry_run:
        for attempt in attempts:
            print(" ".join(attempt_command(args, attempt)))
        _write_summary(output_root, records, None, args.target_accuracy, args.search_metric)
        return
    winner = first_successful_record(records, threshold=args.target_accuracy, metric_key=args.search_metric)
    for index, attempt in enumerate(attempts, start=1):
        print(f"开始 proposed 搜索 {index}/{len(attempts)}: {attempt.name}", flush=True)
        record = _run_attempt(args, attempt)
        records.append(record)
        winner = first_successful_record(records, threshold=args.target_accuracy, metric_key=args.search_metric)
        _write_summary(output_root, records, winner, args.target_accuracy, args.search_metric)
        print(
            f"本次 {args.search_metric}={record[args.search_metric]:.4f}, "
            f"test_acc={record['test_accuracy']:.4f}, "
            f"test_macro_f1={record['test_macro_f1']:.4f}",
            flush=True,
        )
        if winner is not None:
            print(
                f"搜索指标已达到目标: {args.search_metric}={winner[args.search_metric]:.4f}, "
                f"test_acc={winner['test_accuracy']:.4f}, "
                f"metrics={winner['metrics_path']}",
                flush=True,
            )
            return
    _write_summary(output_root, records, winner, args.target_accuracy, args.search_metric)
    raise SystemExit(f"未达到目标搜索指标 {args.search_metric}>={args.target_accuracy:.4f}，已完成 {len(records)} 个候选/尝试")


if __name__ == "__main__":
    main()
