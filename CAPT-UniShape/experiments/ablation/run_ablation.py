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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_train = importlib.import_module("train")
load_config = _train.load_config
run_training = _train.run_training

DEFAULT_ABLATION_VARIANTS = ["full_rbf", "no_rbf", "no_kan_fusion", "static_prototype", "no_condition_input"]
DEFAULT_FULL_METRICS_PATH = ""

ABLATIONS: dict[str, dict[str, Any]] = {
    "full_rbf": {
        "config": "configs/proposed.yaml",
        "description": "完整模型：官方 UniShape + EIS + 工况 + Residual KAN-Fusion + 动态 RBF 原型",
    },
    "no_rbf": {
        "config": "configs/proposed_no_rbf.yaml",
        "description": "E：去掉 RBF 动态原型头，使用 MLP 分类器",
    },
    "fixed_equal_fusion": {
        "config": "configs/proposed.yaml",
        "overrides": {"use_condition_gating": False},
        "description": "B：去掉工况门控 g_op/g_eis，改为固定等权直接相加",
    },
    "no_kan_fusion": {
        "config": "configs/proposed.yaml",
        "overrides": {"use_residual_kan_fusion": False},
        "description": "C：关闭 Residual KAN 分支，只保留融合 MLP",
    },
    "no_film": {
        "config": "configs/proposed.yaml",
        "overrides": {"use_film_modulation": False},
        "description": "D：关闭 FiLM 工况调制，固定 gamma=1、beta=0",
    },
    "static_prototype": {
        "config": "configs/proposed.yaml",
        "overrides": {"use_condition_transport": False},
        "description": "关闭工况感知 prototype transport，只使用静态原型",
    },
    "no_transport_reg": {
        "config": "configs/proposed.yaml",
        "overrides": {"alpha_transport": 0.0},
        "description": "去掉原型迁移幅值正则",
    },
    "no_separation_reg": {
        "config": "configs/proposed.yaml",
        "overrides": {"alpha_sep": 0.0},
        "description": "去掉原型分离正则",
    },
    "no_eis_input": {
        "config": "configs/proposed.yaml",
        "data_zero": ["x_eis"],
        "description": "置零 EIS 分支，验证 EIS 输入贡献",
    },
    "no_condition_input": {
        "config": "configs/proposed.yaml",
        "data_zero": ["x_cond"],
        "description": "置零工况向量，验证工况建模贡献",
    },
    "stack_only": {
        "config": "configs/proposed.yaml",
        "data_zero": ["x_eis", "x_cond"],
        "description": "仅保留电堆运行分支",
    },
    "eis_cond_only": {
        "config": "configs/proposed.yaml",
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
    class0 = test_payload.get("classification_report", {}).get("0", {})
    return {
        "variant": variant,
        "description": description,
        "val_accuracy": float(payload.get("accuracy", 0.0)),
        "val_macro_f1": float(payload.get("macro_f1", 0.0)),
        "val_weighted_f1": float(payload.get("weighted_f1", 0.0)),
        "test_accuracy": float(test_payload.get("accuracy", 0.0)),
        "test_macro_f1": float(test_payload.get("macro_f1", 0.0)),
        "test_weighted_f1": float(test_payload.get("weighted_f1", 0.0)),
        "class0_precision": float(class0.get("precision", 0.0)),
        "class0_recall": float(class0.get("recall", 0.0)),
        "class0_f1": float(class0.get("f1-score", 0.0)),
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
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()

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
