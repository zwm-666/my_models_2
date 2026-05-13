"""Run public AC voltage response baseline experiments and aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_ac_voltage_npz import DOMAIN_MAP, build_ac_voltage_npz


RATIO_TO_TEST_SIZE = {
    "8_2": 0.20,
    "7_3": 0.30,
    "6_4": 0.40,
    "5_5": 0.50,
}

PROTOCOL_DISPLAY_NAME = {
    "old_to_new": "old->new",
    "new_to_old": "new->old",
}

MODEL_CATEGORIES = {
    "proposed": "proposed",
    "logreg": "traditional_ml",
    "svm": "traditional_ml",
    "random_forest": "traditional_ml",
    "mlp": "deep_learning",
    "cnn1d": "deep_learning",
    "lstm": "deep_learning",
    "transformer": "transformer",
    "itransformer": "itransformer",
}


def _baseline_module() -> Any:
    from scripts import run_official_baseline_experiments as baseline

    return baseline


def resolve_public_experiments(protocol: str, ratios: list[str]) -> list[tuple[str, str]]:
    if protocol in PROTOCOL_DISPLAY_NAME:
        return [(protocol, protocol)]
    return [(ratio, ratio) for ratio in ratios]


def public_metric_row(ratio: str, model_key: str, metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_payload = payload.get("test", payload)
    test_source = "test" if "test" in payload else "top_level_fallback"
    return {
        "dataset": "公开数据集",
        "ratio": PROTOCOL_DISPLAY_NAME.get(ratio, ratio.replace("_", ":")),
        "model": model_key,
        "category": MODEL_CATEGORIES[model_key],
        "val_accuracy": float(payload.get("accuracy", 0.0)),
        "val_macro_f1": float(payload.get("macro_f1", 0.0)),
        "test_accuracy": float(test_payload.get("accuracy", 0.0)),
        "test_macro_f1": float(test_payload.get("macro_f1", 0.0)),
        "test_weighted_f1": float(test_payload.get("weighted_f1", 0.0)),
        "test_inference_ms": float(test_payload.get("inference_time_per_sample_ms", 0.0)),
        "test_source": test_source,
        "parameter_count": int(payload.get("parameter_count", 0)),
        "metrics_path": str(metrics_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行公开 AC Voltage Response Data 数据集对比模型实验")
    parser.add_argument("--data-root", default="data/AC Voltage Response Data")
    parser.add_argument("--processed-root", default="data/processed/public_ac_voltage_baselines")
    parser.add_argument("--output-root", default="results/public_ac_voltage_baselines")
    parser.add_argument("--ratios", nargs="+", default=["8_2", "7_3", "6_4", "5_5"], choices=list(RATIO_TO_TEST_SIZE.keys()))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["proposed", "logreg", "svm", "random_forest", "mlp", "cnn1d", "lstm", "transformer", "itransformer"],
        choices=list(MODEL_CATEGORIES.keys()),
    )
    parser.add_argument("--protocol", choices=["old_to_new", "new_to_old", "mixed_stratified"], default="mixed_stratified")
    parser.add_argument("--domain-filter", choices=sorted(DOMAIN_MAP), default=None)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--op-seq-len", type=int, default=256)
    parser.add_argument("--eis-seq-len", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--refit-trainval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-epochs-before-stop", type=int, default=10)
    parser.add_argument("--val-metric-smoothing", type=int, default=1)
    parser.add_argument(
        "--class-weighting",
        choices=["sqrt_balanced", "balanced", "inverse_frequency", "effective_number", "none"],
        default="sqrt_balanced",
    )
    parser.add_argument("--proposed-config", default="configs/rbf_kanfusion.yaml")
    parser.add_argument("--checkpoint-selection", choices=["best_val", "best-val", "last"], default="best_val")
    parser.add_argument("--selection-score", choices=["macro_f1", "macro_f1_class0_recall", "macro_f1_gap_penalty", "val_train_holdout_macro_f1"], default="macro_f1")
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args(argv)


def _build_public_npz(
    data_root: Path,
    processed_root: Path,
    ratio: str,
    seed: int,
    protocol: str,
    op_seq_len: int,
    eis_seq_len: int,
    val_fraction: float,
    domain_filter: str | None = None,
) -> Path:
    domain_part = f"{domain_filter}_" if domain_filter else ""
    output_path = processed_root / f"public_ac_voltage_{domain_part}{protocol}_seed{seed}_{ratio}.npz"
    test_fraction = RATIO_TO_TEST_SIZE.get(ratio, RATIO_TO_TEST_SIZE["8_2"])
    build_ac_voltage_npz(
        data_root=data_root,
        output=output_path,
        protocol=protocol,
        op_seq_len=op_seq_len,
        eis_seq_len=eis_seq_len,
        val_fraction=val_fraction,
        seed=seed,
        test_fraction=test_fraction,
        domain_filter=domain_filter,
    )
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = ROOT / str(args.data_root)
    processed_root = ROOT / str(args.processed_root)
    output_root = ROOT / str(args.output_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    ml_models = {"logreg", "svm", "random_forest"}
    torch_models = {"mlp", "cnn1d", "lstm", "transformer", "itransformer"}

    experiments = resolve_public_experiments(str(args.protocol), [str(item) for item in args.ratios])
    for ratio, build_ratio in experiments:
        npz_path = _build_public_npz(
            data_root=data_root,
            processed_root=processed_root,
            ratio=build_ratio,
            seed=int(args.seed),
            protocol=str(args.protocol),
            op_seq_len=int(args.op_seq_len),
            eis_seq_len=int(args.eis_seq_len),
            val_fraction=float(args.val_fraction),
            domain_filter=args.domain_filter,
        )
        for model_key in args.models:
            baseline = _baseline_module()
            run_dir = output_root / ratio / model_key
            print(f"\n=== 公开数据集对比实验: {model_key} | protocol={PROTOCOL_DISPLAY_NAME.get(ratio, ratio.replace('_', ':'))} ===", flush=True)
            if model_key == "proposed":
                metrics_path = baseline.run_proposed_model(
                    npz_path=npz_path,
                    output_dir=run_dir,
                    config_path=ROOT / str(args.proposed_config),
                    epochs=int(args.epochs),
                    patience=int(args.patience),
                    min_delta=float(args.min_delta),
                    batch_size=int(args.batch_size),
                    refit_trainval=bool(args.refit_trainval),
                    min_epochs_before_stop=int(args.min_epochs_before_stop),
                    val_metric_smoothing=int(args.val_metric_smoothing),
                    class_weighting=str(args.class_weighting),
                    seed=int(args.seed),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    checkpoint_selection=str(args.checkpoint_selection),
                    selection_score=str(args.selection_score),
                    init_checkpoint=args.init_checkpoint,
                )
            elif model_key in ml_models:
                metrics_path = baseline.run_ml_baseline(
                    model_key=model_key,
                    npz_path=npz_path,
                    output_dir=run_dir,
                    seed=int(args.seed),
                    rf_estimators=int(args.rf_estimators),
                    refit_trainval=bool(args.refit_trainval),
                )
            elif model_key in torch_models:
                metrics_path = baseline.run_torch_baseline(
                    model_key=model_key,
                    npz_path=npz_path,
                    output_dir=run_dir,
                    epochs=int(args.epochs),
                    patience=int(args.patience),
                    min_delta=float(args.min_delta),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    seed=int(args.seed),
                    hidden_dim=int(args.hidden_dim),
                    d_model=int(args.d_model),
                    num_layers=int(args.num_layers),
                    dropout=float(args.dropout),
                    refit_trainval=bool(args.refit_trainval),
                    min_epochs_before_stop=int(args.min_epochs_before_stop),
                    val_metric_smoothing=int(args.val_metric_smoothing),
                    class_weighting=str(args.class_weighting),
                )
            else:
                raise KeyError(model_key)
            row = public_metric_row(ratio, model_key, metrics_path)
            rows.append(row)
            baseline._print_metric_row(row)

    summary_path = output_root / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    test_summary_path = output_root / "test_summary.csv"
    test_fieldnames = [
        "dataset",
        "ratio",
        "model",
        "category",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_inference_ms",
        "parameter_count",
        "test_source",
        "metrics_path",
    ]
    with test_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=test_fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in test_fieldnames} for row in rows])

    print(f"\n已写入公开数据集对比实验汇总: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
