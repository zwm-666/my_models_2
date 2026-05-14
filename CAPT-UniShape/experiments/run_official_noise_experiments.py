"""Run one-ratio clean-training/noisy-test robustness experiments."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_converter = importlib.import_module("experiments.build_official_npz_from_self_excel")
_train = importlib.import_module("train")
_models = importlib.import_module("models")
build_npz = _converter.build_npz
FuelCellNPZDataset = _train.FuelCellNPZDataset
count_parameters = _train.count_parameters
evaluate_loader = _train.evaluate_loader
load_config = _train.load_config
run_training = _train.run_training
save_outputs = _train.save_outputs
build_model_from_config = _models.build_model_from_config


RATIO_TO_TEST_SIZE = {
    "8_2": 0.20,
    "7_3": 0.30,
    "6_4": 0.40,
    "5_5": 0.50,
}

MODELS = {
    "rbf": "configs/rbf_kanfusion.yaml",
    "no_rbf": "configs/kanfusion_no_rbf.yaml",
}


def _subset_and_noise_test_npz(
    base_npz: Path,
    output_npz: Path,
    noise_std: float,
    noise_targets: list[str],
    seed: int,
) -> Path:
    data = np.load(base_npz)
    if "split" not in data:
        raise ValueError(f"{base_npz} 缺少 split 数组，无法只对测试集做噪声实验")

    split = np.asarray(data["split"], dtype=np.int64)
    test_idx = np.where(split == 2)[0]
    if len(test_idx) == 0:
        raise ValueError(f"{base_npz} 没有 split=2 的测试样本")

    n_samples = int(split.shape[0])
    payload: dict[str, np.ndarray[Any, Any]] = {}
    for key in data.files:
        array = np.asarray(data[key])
        if array.ndim > 0 and int(array.shape[0]) == n_samples:
            payload[key] = array[test_idx].copy()
        else:
            payload[key] = array.copy()

    rng = np.random.default_rng(seed)
    for key in noise_targets:
        if key not in payload:
            raise KeyError(f"{base_npz} 中不存在可加噪字段 {key}")
        array = payload[key].astype(np.float32, copy=True)
        scale = float(np.std(array))
        if scale < 1e-8:
            scale = 1.0
        noise = rng.normal(loc=0.0, scale=noise_std * scale, size=array.shape).astype(np.float32)
        payload[key] = array + noise

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    return output_npz


def _evaluate_checkpoint(config: dict[str, Any], data_path: Path, checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    dataset = FuelCellNPZDataset(data_path)
    batch_size = int(config.get("experiment", {}).get("batch_size", 8))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    metrics = evaluate_loader(model, loader, device)
    save_outputs(output_dir, metrics, count_parameters(model))
    return metrics


def _load_checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise KeyError(f"{checkpoint_path} 中缺少训练时保存的 config")
    return config


def _metric_row(
    ratio: str,
    model_key: str,
    noise_std: float,
    noise_targets: list[str],
    data_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "ratio": ratio.replace("_", ":"),
        "model": model_key,
        "noise_std": float(noise_std),
        "noise_targets": "+".join(noise_targets),
        "test_accuracy": float(payload.get("accuracy", 0.0)),
        "test_macro_f1": float(payload.get("macro_f1", 0.0)),
        "test_weighted_f1": float(payload.get("weighted_f1", 0.0)),
        "test_inference_ms": float(payload.get("inference_time_per_sample_ms", 0.0)),
        "parameter_count": int(payload.get("parameter_count", 0)),
        "data_path": str(data_path),
        "metrics_path": str(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行单一训练/测试比例下的噪声鲁棒性实验")
    parser.add_argument("--excel", default="data/raw/水淹和膜干故障测试数据_补充特征汇总.xlsx")
    parser.add_argument("--ratio", default="8_2", choices=list(RATIO_TO_TEST_SIZE.keys()))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--noise-stds", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--noise-targets", nargs="+", default=["x_op", "x_eis", "x_cond"], choices=["x_op", "x_eis", "x_cond"])
    parser.add_argument("--output-root", default="results/official_noise_experiments")
    parser.add_argument("--data-root", default="data/processed/official_noise_experiments")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride-train", type=int, default=16)
    parser.add_argument("--stride-eval", type=int, default=32)
    parser.add_argument("--eis-seq-len", type=int, default=128)
    parser.add_argument("--split-mode", default="segment")
    parser.add_argument("--segment-gap-seconds", type=float, default=600.0)
    parser.add_argument("--segment-block-seconds", type=float, default=300.0)
    parser.add_argument("--segment-label-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group-split-strategy", choices=["holdout_first", "three_way", "two_stage"], default="holdout_first", help="分组划分策略；holdout_first 先划分训练集与 held-out，再从 held-out 中分出验证/测试")
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--refit-trainval", action=argparse.BooleanOptionalAction, default=False, help="验证集只用于选epoch，最终模型用train+val重训；默认关闭以保持验证/测试独立")
    parser.add_argument("--min-epochs-before-stop", type=int, default=20)
    parser.add_argument("--val-metric-smoothing", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-aware-train-stride", action="store_true", help="训练集按类别自动调节滑窗步长：少数类更密，多数类更稀")
    parser.add_argument("--min-train-stride", type=int, default=None, help="类别感知训练步长下限；默认 stride_train//2")
    parser.add_argument("--max-train-stride", type=int, default=None, help="类别感知训练步长上限；默认 stride_train*2")
    parser.add_argument("--class-stride-power", type=float, default=1.0, help="类别样本数到训练步长的缩放幂指数")
    parser.add_argument("--split-retries", type=int, default=50, help="重试分组划分并选择少数类支持更好的 split")
    parser.add_argument("--min-eval-class-windows", type=int, default=5, help="验证/测试集中任一类别窗口数低于该值时标记为不稳定")
    parser.add_argument("--min-eval-class-groups", type=int, default=1, help="验证/测试集中任一类别 group 数低于该值时标记为不稳定")
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    data_root = ROOT / args.data_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    clean_npz = data_root / f"official_self_stack_impedance_eis_w{args.window_size}_{args.ratio}_clean.npz"
    print(f"\n=== 构建干净数据: train/test={args.ratio.replace('_', ':')} ===", flush=True)
    build_npz(
        excel_path=ROOT / args.excel,
        output_path=clean_npz,
        window_size=args.window_size,
        stride_train=args.stride_train,
        stride_eval=args.stride_eval,
        eis_seq_len=args.eis_seq_len,
        split_mode=args.split_mode,
        segment_gap_seconds=args.segment_gap_seconds,
        segment_block_seconds=args.segment_block_seconds,
        segment_label_boundary=args.segment_label_boundary,
        random_state=args.seed,
        op_source="stack",
        test_size=RATIO_TO_TEST_SIZE[args.ratio],
        val_size=args.val_size,
        class_aware_train_stride=args.class_aware_train_stride,
        min_train_stride=args.min_train_stride,
        max_train_stride=args.max_train_stride,
        class_stride_power=args.class_stride_power,
        group_split_strategy=args.group_split_strategy,
        split_retries=args.split_retries,
        min_eval_class_windows=args.min_eval_class_windows,
        min_eval_class_groups=args.min_eval_class_groups,
    )

    rows: list[dict[str, Any]] = []
    noise_targets = list(args.noise_targets)
    for model_key in args.models:
        config_path = MODELS[model_key]
        train_dir = output_root / args.ratio / model_key / "clean_train"
        config = load_config(ROOT / config_path)
        config.setdefault("experiment", {})["output_dir"] = str(train_dir)
        print(f"\n=== 干净训练 {model_key} | train/test={args.ratio.replace('_', ':')} ===", flush=True)
        run_training(
            config=config,
            data_path=clean_npz,
            output_dir=train_dir,
            epochs_override=args.epochs,
            patience_override=args.patience,
            min_delta_override=args.min_delta,
            refit_trainval_override=args.refit_trainval,
            min_epochs_before_stop_override=args.min_epochs_before_stop,
            val_metric_smoothing_override=args.val_metric_smoothing,
        )
        checkpoint_path = train_dir / "best.ckpt"
        trained_config = _load_checkpoint_config(checkpoint_path)

        for noise_std in args.noise_stds:
            noise_name = f"noise_{noise_std:.3f}".replace(".", "p")
            noisy_npz = data_root / args.ratio / model_key / f"{noise_name}.npz"
            print(f"\n=== 噪声测试 {model_key} | std={noise_std:.3f} | targets={'+'.join(noise_targets)} ===", flush=True)
            _subset_and_noise_test_npz(
                base_npz=clean_npz,
                output_npz=noisy_npz,
                noise_std=float(noise_std),
                noise_targets=noise_targets,
                seed=int(args.seed) + int(round(float(noise_std) * 10000)),
            )
            eval_dir = output_root / args.ratio / model_key / noise_name
            _evaluate_checkpoint(trained_config, noisy_npz, checkpoint_path, eval_dir)
            rows.append(_metric_row(args.ratio, model_key, float(noise_std), noise_targets, noisy_npz, eval_dir / "metrics.json"))

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已写入噪声鲁棒性实验汇总: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

