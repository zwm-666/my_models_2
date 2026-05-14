"""Run official CAPT-UniShape ablation experiments with early stopping."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_train = importlib.import_module("train")
_evaluate = importlib.import_module("evaluate")
_snr = importlib.import_module("experiments.refresh_snr_noise_results")
load_config = _train.load_config
run_training = _train.run_training
run_evaluation = _evaluate.run_evaluation
write_test_subset_with_snr_noise = _snr.write_test_subset_with_snr_noise
noise_seed_for_snr = _snr.noise_seed_for_snr

DEFAULT_ABLATION_VARIANTS = ["full_rbf", "no_rbf", "no_kan_fusion", "static_prototype", "no_condition_input"]
DEFAULT_FULL_METRICS_PATH = ""
DEFAULT_ABLATION_SNR_DBS = [30.0, 20.0, 10.0]
DEFAULT_SNR_NOISE_SEEDS = [44, 45, 46]
DEFAULT_NOISE_TARGETS = ["x_op", "x_eis", "x_cond"]
ABLATION_CONFIG = "configs/ablation.yaml"

ABLATIONS: dict[str, dict[str, Any]] = {
    "full_rbf": {
        "config": ABLATION_CONFIG,
        "description": "完整模型：官方 UniShape + EIS + 工况 + Residual KAN-Fusion + 动态 RBF 原型",
    },
    "no_rbf": {
        "config": ABLATION_CONFIG,
        "overrides": {"model_name": "official_capt_unishape_kanfusion_no_rbf", "use_rbf_head": False},
        "description": "E：去掉 RBF 动态原型头，使用 MLP 分类器",
    },
    "fixed_equal_fusion": {
        "config": ABLATION_CONFIG,
        "overrides": {"use_condition_gating": False},
        "description": "B：去掉工况门控 g_op/g_eis，改为固定等权直接相加",
    },
    "no_kan_fusion": {
        "config": ABLATION_CONFIG,
        "overrides": {"use_residual_kan_fusion": False},
        "description": "C：关闭 Residual KAN 分支，只保留融合 MLP",
    },
    "no_film": {
        "config": ABLATION_CONFIG,
        "overrides": {"use_film_modulation": False},
        "description": "D：关闭 FiLM 工况调制，固定 gamma=1、beta=0",
    },
    "static_prototype": {
        "config": ABLATION_CONFIG,
        "overrides": {"use_condition_transport": False},
        "description": "关闭工况感知 prototype transport，只使用静态原型",
    },
    "no_transport_reg": {
        "config": ABLATION_CONFIG,
        "overrides": {"alpha_transport": 0.0},
        "description": "去掉原型迁移幅值正则",
    },
    "no_separation_reg": {
        "config": ABLATION_CONFIG,
        "overrides": {"alpha_sep": 0.0},
        "description": "去掉原型分离正则",
    },
    "no_eis_input": {
        "config": ABLATION_CONFIG,
        "data_zero": ["x_eis"],
        "description": "置零 EIS 分支，验证 EIS 输入贡献",
    },
    "no_condition_input": {
        "config": ABLATION_CONFIG,
        "data_zero": ["x_cond"],
        "description": "置零工况向量，验证工况建模贡献",
    },
    "stack_only": {
        "config": ABLATION_CONFIG,
        "data_zero": ["x_eis", "x_cond"],
        "description": "仅保留电堆运行分支",
    },
    "eis_cond_only": {
        "config": ABLATION_CONFIG,
        "data_zero": ["x_op"],
        "description": "去掉电堆运行分支，仅保留 EIS + 工况",
    },
}


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _make_data_variant(base_npz: Path, output_npz: Path, zero_keys: list[str]) -> Path:
    if not zero_keys:
        return base_npz
    data = np.load(base_npz)
    payload: dict[str, np.ndarray[Any, Any]] = {key: data[key].copy() for key in data.files}
    for key in zero_keys:
        if key not in payload:
            raise KeyError(f"{base_npz} 中不存在 {key}")
        payload[key] = np.zeros_like(payload[key])
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    return output_npz


def _metric_row(variant: str, description: str, metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_payload = payload.get("test", payload)
    return {
        "variant": variant,
        "description": description,
        "val_accuracy": float(payload.get("accuracy", 0.0)),
        "val_macro_f1": float(payload.get("macro_f1", 0.0)),
        "val_weighted_f1": float(payload.get("weighted_f1", 0.0)),
        "test_accuracy": float(test_payload.get("accuracy", 0.0)),
        "test_macro_f1": float(test_payload.get("macro_f1", 0.0)),
        "test_weighted_f1": float(test_payload.get("weighted_f1", 0.0)),
        "test_inference_ms": float(test_payload.get("inference_time_per_sample_ms", 0.0)),
        "parameter_count": int(payload.get("parameter_count", 0)),
        "metrics_path": str(metrics_path),
    }


def _copy_existing_metric_row(variant: str, description: str, metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"full_rbf 复用结果不存在: {metrics_path}")
    return _metric_row(variant, description, metrics_path)


def _format_summary_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pct(value: Any) -> str:
    return f"{100.0 * _safe_float(value):.2f}%"


def _snr_token(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _variant_overrides(spec: dict[str, Any]) -> dict[str, Any]:
    return dict(spec.get("overrides", {}))


def _active_noise_targets(spec: dict[str, Any], requested_targets: list[str]) -> list[str]:
    zero_keys = set(str(key) for key in spec.get("data_zero", []))
    return [target for target in requested_targets if target not in zero_keys]


def _load_test_payload(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_payload = payload.get("test")
    if isinstance(test_payload, dict):
        return test_payload
    return payload


def _snr_result_row(
    *,
    variant: str,
    description: str,
    snr_db: str,
    noise_seed: Any,
    n_noise_seeds: Any,
    actual_snr_db_mean: Any,
    noise_targets: list[str],
    zeroed_inputs: list[str],
    data_path: Path,
    metrics_path: Path,
    clean_metrics: dict[str, Any],
) -> dict[str, Any]:
    metrics = _load_test_payload(metrics_path)
    test_accuracy = float(metrics.get("accuracy", 0.0))
    test_macro_f1 = float(metrics.get("macro_f1", 0.0))
    test_weighted_f1 = float(metrics.get("weighted_f1", 0.0))
    clean_accuracy = float(clean_metrics.get("accuracy", 0.0))
    clean_macro_f1 = float(clean_metrics.get("macro_f1", 0.0))
    clean_weighted_f1 = float(clean_metrics.get("weighted_f1", 0.0))
    return {
        "variant": variant,
        "description": description,
        "snr_db": snr_db,
        "noise_seed": noise_seed,
        "n_noise_seeds": n_noise_seeds,
        "actual_snr_db_mean": actual_snr_db_mean,
        "noise_targets": "+".join(noise_targets),
        "zeroed_inputs": "+".join(zeroed_inputs),
        "test_accuracy": test_accuracy,
        "accuracy_drop": clean_accuracy - test_accuracy,
        "test_macro_f1": test_macro_f1,
        "macro_f1_drop": clean_macro_f1 - test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "weighted_f1_drop": clean_weighted_f1 - test_weighted_f1,
        "test_inference_ms": float(metrics.get("inference_time_per_sample_ms", 0.0)),
        "parameter_count": int(json.loads(metrics_path.read_text(encoding="utf-8")).get("parameter_count", 0)),
        "data_path": str(data_path),
        "metrics_path": str(metrics_path),
    }


def _paper_snr_row(row: dict[str, Any]) -> dict[str, Any]:
    actual_snr = row.get("actual_snr_db_mean", "")
    return {
        "variant": row["variant"],
        "description": row["description"],
        "snr_db": row["snr_db"],
        "noise_seed": row.get("noise_seed", ""),
        "n_noise_seeds": row.get("n_noise_seeds", ""),
        "actual_snr_db_mean": "" if actual_snr == "" else f"{_safe_float(actual_snr):.4f}",
        "noise_targets": row["noise_targets"],
        "zeroed_inputs": row["zeroed_inputs"],
        "Acc": _pct(row["test_accuracy"]),
        "Acc_drop": _pct(row["accuracy_drop"]),
        "Macro-F1": _pct(row["test_macro_f1"]),
        "Macro-F1_drop": _pct(row["macro_f1_drop"]),
        "Weighted-F1": _pct(row["test_weighted_f1"]),
        "Weighted-F1_drop": _pct(row["weighted_f1_drop"]),
        "test_inference_ms": f"{_safe_float(row['test_inference_ms']):.4f}",
        "parameter_count": row["parameter_count"],
        "metrics_path": row["metrics_path"],
    }


def _mean_snr_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    group_order: list[tuple[str, str]] = []
    for row in rows:
        if str(row.get("snr_db", "")) == "clean":
            output.append(dict(row))
            continue
        key = (str(row.get("variant", "")), str(row.get("snr_db", "")))
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(row)

    mean_fields = [
        "actual_snr_db_mean",
        "test_accuracy",
        "accuracy_drop",
        "test_macro_f1",
        "macro_f1_drop",
        "test_weighted_f1",
        "weighted_f1_drop",
        "test_inference_ms",
    ]
    for key in group_order:
        group = grouped[key]
        first = dict(group[0])
        for field in mean_fields:
            first[field] = sum(_safe_float(row.get(field)) for row in group) / len(group)
        first["noise_seed"] = "mean"
        first["n_noise_seeds"] = len(group)
        first["data_path"] = "mean_of_noise_seed_npzs"
        first["metrics_path"] = "mean_of_noise_seed_metrics"
        output.append(first)
    return output


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_snr_ablation_evaluation(args: argparse.Namespace) -> None:
    base_npz = ROOT / args.data
    output_root = ROOT / (args.snr_output_root or args.output_root)
    data_root = ROOT / (args.snr_data_root or args.data_root)
    checkpoint_root = ROOT / args.reuse_checkpoints_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    requested_targets = list(args.noise_targets)
    for variant in args.variants:
        spec = ABLATIONS[variant]
        description = str(spec["description"])
        zero_keys = list(spec.get("data_zero", []))
        variant_clean_npz = _make_data_variant(base_npz, data_root / variant / "clean.npz", zero_keys)
        checkpoint_path = checkpoint_root / variant / "best.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{variant} 缺少 checkpoint: {checkpoint_path}")

        clean_eval_dir = output_root / variant / "clean"
        print(f"\n=== SNR 消融评估: {variant} clean ===", flush=True)
        run_evaluation(
            str(ROOT / str(spec["config"])),
            str(variant_clean_npz),
            str(checkpoint_path),
            str(clean_eval_dir),
            split="test",
            config_overrides=_variant_overrides(spec),
        )
        clean_metrics_path = clean_eval_dir / "metrics.json"
        clean_metrics = _load_test_payload(clean_metrics_path)
        rows.append(
            _snr_result_row(
                variant=variant,
                description=description,
                snr_db="clean",
                noise_seed="",
                n_noise_seeds="",
                actual_snr_db_mean="",
                noise_targets=_active_noise_targets(spec, requested_targets),
                zeroed_inputs=zero_keys,
                data_path=variant_clean_npz,
                metrics_path=clean_metrics_path,
                clean_metrics=clean_metrics,
            )
        )

        active_targets = _active_noise_targets(spec, requested_targets)
        if not active_targets:
            raise ValueError(f"{variant} 没有可加噪输入；requested_targets={requested_targets}, zero_keys={zero_keys}")
        for snr_db in args.snr_dbs:
            label = _snr_token(float(snr_db))
            for noise_seed in args.snr_noise_seeds:
                noisy_npz = data_root / variant / f"seed_{int(noise_seed)}" / f"snr_{label}dB.npz"
                noise_summary = write_test_subset_with_snr_noise(
                    variant_clean_npz,
                    noisy_npz,
                    snr_db=float(snr_db),
                    noise_targets=active_targets,
                    seed=noise_seed_for_snr(int(noise_seed), float(snr_db)),
                )
                eval_dir = output_root / variant / f"seed_{int(noise_seed)}" / f"snr_{label}dB"
                print(f"=== SNR 消融评估: {variant} {label}dB seed={int(noise_seed)} ===", flush=True)
                run_evaluation(
                    str(ROOT / str(spec["config"])),
                    str(noisy_npz),
                    str(checkpoint_path),
                    str(eval_dir),
                    split="test",
                    config_overrides=_variant_overrides(spec),
                )
                rows.append(
                    _snr_result_row(
                        variant=variant,
                        description=description,
                        snr_db=label,
                        noise_seed=int(noise_seed),
                        n_noise_seeds=1,
                        actual_snr_db_mean=float(noise_summary["actual_snr_db_mean"]),
                        noise_targets=active_targets,
                        zeroed_inputs=zero_keys,
                        data_path=noisy_npz,
                        metrics_path=eval_dir / "metrics.json",
                        clean_metrics=clean_metrics,
                    )
                )

    detail_summary_path = output_root / "summary_by_seed.csv"
    summary_path = output_root / "summary.csv"
    paper_summary_path = ROOT / args.snr_summary_path
    mean_rows = _mean_snr_rows(rows)
    _write_rows(detail_summary_path, [{key: _format_summary_value(value) for key, value in row.items()} for row in rows])
    _write_rows(summary_path, [{key: _format_summary_value(value) for key, value in row.items()} for row in mean_rows])
    _write_rows(paper_summary_path, [_paper_snr_row(row) for row in mean_rows])
    print(f"\n已写入逐 seed SNR 消融汇总: {detail_summary_path}", flush=True)
    print(f"已写入 3 seed 均值 SNR 消融汇总: {summary_path}", flush=True)
    print(f"已写入论文口径 3 seed 均值 SNR 消融表: {paper_summary_path}", flush=True)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行官方 CAPT-UniShape 消融实验")
    parser.add_argument("--data", default="data/processed/self_seed44_8_2.npz")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_ABLATION_VARIANTS, choices=list(ABLATIONS.keys()))
    parser.add_argument("--output-root", default="results/official_ablation")
    parser.add_argument("--data-root", default="data/processed/official_ablation")
    parser.add_argument("--full-metrics-path", default=DEFAULT_FULL_METRICS_PATH, help="full_rbf 直接复用的已有 metrics.json；设为空字符串则重新训练 full_rbf")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--class-weighting", choices=["sqrt_balanced", "balanced", "inverse_frequency", "effective_number", "balanced_softmax", "logit_adjusted", "none"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--checkpoint-selection", choices=["best_val", "best-val", "last"], default=None)
    parser.add_argument("--refit-trainval", action=argparse.BooleanOptionalAction, default=None, help="验证集只用于选epoch，最终模型用train+val重训；默认使用配置文件设置")
    parser.add_argument("--min-epochs-before-stop", type=int, default=None)
    parser.add_argument("--val-metric-smoothing", type=int, default=None)
    parser.add_argument("--snr-eval-only", action="store_true", help="复用已有消融 checkpoint，只评估 clean 与 SNR 噪声测试集")
    parser.add_argument("--reuse-checkpoints-root", default="results/current_ablation_updated_dataset_seed44_6_4")
    parser.add_argument("--snr-output-root", default="results/current_ablation_snr_updated_dataset_seed44_6_4")
    parser.add_argument("--snr-data-root", default="data/processed/current_ablation_snr_updated_dataset_seed44_6_4")
    parser.add_argument("--snr-summary-path", default="results/消融实验SNR新表.csv")
    parser.add_argument("--snr-dbs", nargs="+", type=float, default=DEFAULT_ABLATION_SNR_DBS)
    parser.add_argument("--noise-targets", nargs="+", default=DEFAULT_NOISE_TARGETS, choices=DEFAULT_NOISE_TARGETS)
    parser.add_argument("--snr-noise-seeds", nargs="+", type=int, default=DEFAULT_SNR_NOISE_SEEDS)
    parser.add_argument("--snr-seed", type=int, default=44, help=argparse.SUPPRESS)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.snr_eval_only:
        run_snr_ablation_evaluation(args)
        return

    base_npz = ROOT / args.data
    output_root = ROOT / args.output_root
    data_root = ROOT / args.data_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        spec = ABLATIONS[variant]
        if variant == "full_rbf" and args.full_metrics_path:
            metrics_path = _resolve_path(args.full_metrics_path)
            print(f"\n=== 消融实验: {variant} ===", flush=True)
            print(f"{spec['description']}；复用已有结果: {metrics_path}", flush=True)
            rows.append(_copy_existing_metric_row(variant, str(spec["description"]), metrics_path))
            continue
        zero_keys = list(spec.get("data_zero", []))
        data_path = _make_data_variant(base_npz, data_root / f"{variant}.npz", zero_keys)
        config = load_config(ROOT / str(spec["config"]))
        for key, value in dict(spec.get("overrides", {})).items():
            config[key] = value
        run_dir = output_root / variant
        experiment = config.setdefault("experiment", {})
        experiment["output_dir"] = str(run_dir)
        if args.seed is not None:
            experiment["seeds"] = [int(args.seed)]
        print(f"\n=== 消融实验: {variant} ===", flush=True)
        print(spec["description"], flush=True)
        run_training(
            config=config,
            data_path=data_path,
            output_dir=run_dir,
            epochs_override=args.epochs,
            patience_override=args.patience,
            min_delta_override=args.min_delta,
            batch_size_override=args.batch_size,
            refit_trainval_override=args.refit_trainval,
            min_epochs_before_stop_override=args.min_epochs_before_stop,
            val_metric_smoothing_override=args.val_metric_smoothing,
            class_weighting_override=args.class_weighting,
            lr_override=args.lr,
            weight_decay_override=args.weight_decay,
            checkpoint_selection_override=args.checkpoint_selection,
        )
        rows.append(_metric_row(variant, str(spec["description"]), run_dir / "metrics.json"))

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([{key: _format_summary_value(value) for key, value in row.items()} for row in rows])
    print(f"\n已写入消融实验汇总: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

