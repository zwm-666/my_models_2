"""Run official baseline experiments over multiple seeds and aggregate results."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = ROOT / "scripts" / "run_official_baseline_experiments.py"
RATIO_KEYS = ["8_2", "7_3", "6_4", "5_5"]
MODEL_KEYS = ["proposed", "logreg", "svm", "random_forest", "mlp", "cnn1d", "lstm", "transformer", "itransformer"]


def _child_path(root: str, seed: int) -> str:
    return str(Path(root) / f"seed_{seed}")


def _append_optional(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def _baseline_command(args: argparse.Namespace, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--excel",
        str(args.excel),
        "--ratios",
        *args.ratios,
        "--models",
        *args.models,
        "--output-root",
        _child_path(str(args.output_root), seed),
        "--data-root",
        _child_path(str(args.data_root), seed),
        "--seed",
        str(seed),
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
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-dim",
        str(args.hidden_dim),
        "--d-model",
        str(args.d_model),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--rf-estimators",
        str(args.rf_estimators),
        "--class-weighting",
        str(args.class_weighting),
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
    if not bool(args.segment_label_boundary):
        command.append("--no-segment-label-boundary")
    if bool(args.refit_trainval):
        command.append("--refit-trainval")
    else:
        command.append("--no-refit-trainval")
    if bool(args.class_aware_train_stride):
        command.append("--class-aware-train-stride")
    _append_optional(command, "--min-train-stride", args.min_train_stride)
    _append_optional(command, "--max-train-stride", args.max_train_stride)
    return command


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _read_seed_summary(path: Path, seed: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing seed test summary: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = str(seed)
    return rows


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("ratio", ""), row.get("model", ""), row.get("category", ""))
        grouped.setdefault(key, []).append(row)
    metric_names = ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_inference_ms"]
    aggregated: list[dict[str, str]] = []
    for (ratio, model, category), group_rows in sorted(grouped.items()):
        output: dict[str, str] = {"ratio": ratio, "model": model, "category": category, "n_seeds": str(len(group_rows))}
        for metric_name in metric_names:
            values = [_safe_float(row.get(metric_name, "")) for row in group_rows]
            output[f"{metric_name}_mean"] = f"{fmean(values):.6f}" if values else ""
            output[f"{metric_name}_std"] = f"{stdev(values):.6f}" if len(values) > 1 else "0.000000"
            output[f"{metric_name}_min"] = f"{min(values):.6f}" if values else ""
            output[f"{metric_name}_max"] = f"{max(values):.6f}" if values else ""
        parameter_values = [_safe_float(row.get("parameter_count", "")) for row in group_rows]
        output["parameter_count"] = str(int(round(fmean(parameter_values)))) if parameter_values else ""
        output["seed_values"] = ";".join(row.get("seed", "") for row in group_rows)
        aggregated.append(output)
    return aggregated


def _run_diagnosis(results_root: str, data_root: str) -> None:
    diagnose_script = ROOT / "scripts" / "diagnose_official_results.py"
    subprocess.run(
        [
            sys.executable,
            str(diagnose_script),
            "--results-root",
            results_root,
            "--data-root",
            data_root,
        ],
        cwd=ROOT,
        check=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行多 seed 官方基准实验并汇总测试集均值/方差")
    parser.add_argument("--excel", default="data/raw/水淹和膜干故障测试数据_补充特征汇总.xlsx")
    parser.add_argument("--ratios", nargs="+", default=RATIO_KEYS, choices=RATIO_KEYS)
    parser.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-root", default="results/official_baseline_multiseed")
    parser.add_argument("--data-root", default="data/processed/official_baseline_multiseed")
    parser.add_argument("--skip-run", action="store_true", help="只聚合已有 seed_* 目录，不重新训练")
    parser.add_argument("--diagnose", action=argparse.BooleanOptionalAction, default=True, help="每个 seed 结束后运行产物诊断")
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
    parser.add_argument("--min-epochs-before-stop", type=int, default=20)
    parser.add_argument("--val-metric-smoothing", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--class-aware-train-stride", action="store_true")
    parser.add_argument("--min-train-stride", type=int, default=None)
    parser.add_argument("--max-train-stride", type=int, default=None)
    parser.add_argument(
        "--class-weighting",
        choices=["sqrt_balanced", "balanced", "inverse_frequency", "effective_number", "balanced_softmax", "logit_adjusted", "none"],
        default="sqrt_balanced",
    )
    parser.add_argument("--proposed-config", default="configs/rbf_kanfusion.yaml", help="提出模型使用的 YAML 配置路径，透传给单 seed 官方基准脚本")
    parser.add_argument("--checkpoint-selection", choices=["best_val", "best-val", "last"], default="best_val")
    parser.add_argument("--split-retries", type=int, default=50)
    parser.add_argument("--min-eval-class-windows", type=int, default=5)
    parser.add_argument("--min-eval-class-groups", type=int, default=1)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_root = ROOT / str(args.output_root)
    data_root = ROOT / str(args.data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    for seed in args.seeds:
        seed_output = _child_path(str(args.output_root), int(seed))
        seed_data = _child_path(str(args.data_root), int(seed))
        if not bool(args.skip_run):
            command = _baseline_command(args, int(seed))
            print("\n=== 多 seed 实验开始 ===", flush=True)
            print(f"seed={seed}", flush=True)
            print(" ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            if bool(args.diagnose):
                _run_diagnosis(seed_output, seed_data)
        all_rows.extend(_read_seed_summary(ROOT / seed_output / "test_summary.csv", int(seed)))
    all_fieldnames = ["seed", "ratio", "model", "category", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_inference_ms", "parameter_count", "metrics_path"]
    _write_rows(output_root / "all_seed_test_summary.csv", all_rows, all_fieldnames)
    aggregated = _aggregate(all_rows)
    aggregate_fieldnames = [
        "ratio",
        "model",
        "category",
        "n_seeds",
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
        "seed_values",
    ]
    _write_rows(output_root / "multiseed_test_summary.csv", aggregated, aggregate_fieldnames)
    ranked = sorted(aggregated, key=lambda row: (row["ratio"], -_safe_float(row["test_macro_f1_mean"]), row["model"]))
    _write_rows(output_root / "ranked_multiseed_test_summary.csv", ranked, aggregate_fieldnames)
    print(f"\n已写入多 seed 明细: {output_root / 'all_seed_test_summary.csv'}", flush=True)
    print(f"已写入多 seed 汇总: {output_root / 'multiseed_test_summary.csv'}", flush=True)
    print(f"已写入多 seed 排名: {output_root / 'ranked_multiseed_test_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
