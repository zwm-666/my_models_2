"""Minimal training entry for official CAPT-UniShape RBF/No-RBF experiments.

Expected NPZ keys for real data: ``x_op`` [N,C_op,T], ``x_eis`` [N,C_eis,F],
``x_cond`` or ``cond`` [N,D_cond], and ``labels`` or ``y`` [N].
"""

from __future__ import annotations

import argparse
import csv
from importlib import import_module
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Protocol, cast, runtime_checkable

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

build_model_from_config = getattr(import_module("models"), "build_model_from_config")


@runtime_checkable
class _ReconfigurableTextIO(Protocol):
    def reconfigure(self, *, encoding: str) -> None:
        ...


@runtime_checkable
class _SizedDataset(Protocol):
    def __len__(self) -> int:
        ...


if isinstance(sys.stdout, _ReconfigurableTextIO):
    sys.stdout.reconfigure(encoding="utf-8")


class FuelCellNPZDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """NPZ-backed dataset for operation/EIS/condition fault classification."""

    def __init__(self, npz_path: str | Path, indices: np.ndarray[Any, Any] | None = None) -> None:
        data = np.load(npz_path)
        indexer = indices if indices is not None else slice(None)
        self.x_op = torch.as_tensor(data["x_op"][indexer], dtype=torch.float32)
        self.x_eis = torch.as_tensor(data["x_eis"][indexer], dtype=torch.float32)
        cond_key = "x_cond" if "x_cond" in data else "cond"
        label_key = "labels" if "labels" in data else "y"
        self.x_cond = torch.as_tensor(data[cond_key][indexer], dtype=torch.float32)
        self.labels = torch.as_tensor(data[label_key][indexer], dtype=torch.long)
        if not (len(self.x_op) == len(self.x_eis) == len(self.x_cond) == len(self.labels)):
            raise ValueError("NPZ arrays must have the same first dimension")

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_op[index], self.x_eis[index], self.x_cond[index], self.labels[index]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def sync_config_with_dataset(config: dict[str, Any], dataset: FuelCellNPZDataset) -> dict[str, Any]:
    updated = dict(config)
    updated["c_op"] = int(dataset.x_op.shape[1])
    updated["op_seq_len"] = int(dataset.x_op.shape[2])
    updated["c_eis"] = int(dataset.x_eis.shape[1])
    updated["eis_seq_len"] = int(dataset.x_eis.shape[2])
    updated["d_cond"] = int(dataset.x_cond.shape[1])
    updated["num_classes"] = int(torch.unique(dataset.labels).numel())
    return updated


def split_dataset(
    dataset: FuelCellNPZDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[
    Subset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    Subset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_size = max(1, int(len(indices) * float(val_ratio)))
    return Subset(dataset, indices[val_size:]), Subset(dataset, indices[:val_size])


def build_datasets_from_npz(
    data_path: str | Path,
    seed: int,
    val_ratio: float,
) -> tuple[
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    FuelCellNPZDataset | None,
]:
    """Build train/val(/test) datasets, respecting an optional NPZ split array."""
    data = np.load(data_path)
    if "split" in data:
        split = np.asarray(data["split"], dtype=np.int64)
        train_idx = np.where(split == 0)[0]
        val_idx = np.where(split == 1)[0]
        test_idx = np.where(split == 2)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("NPZ split array must contain train=0 and val=1 samples")
        test_ds = FuelCellNPZDataset(data_path, test_idx) if len(test_idx) > 0 else None
        return FuelCellNPZDataset(data_path, train_idx), FuelCellNPZDataset(data_path, val_idx), test_ds
    dataset = FuelCellNPZDataset(data_path)
    train_subset, val_subset = split_dataset(dataset, val_ratio=val_ratio, seed=seed)
    return train_subset, val_subset, None


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def set_reproducible_seed(seed: int) -> None:
    """Set the random state used by Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def build_experiment_model(
    config: dict[str, Any],
    device: torch.device,
    op_pretrained: str | None = None,
    eis_pretrained: str | None = None,
) -> torch.nn.Module:
    model_obj = build_model_from_config(config)
    if not isinstance(model_obj, torch.nn.Module):
        raise TypeError("build_model_from_config must return a torch.nn.Module")
    model = model_obj.to(device)
    op_ckpt = op_pretrained or config.get("op_pretrained_checkpoint") or config.get("pretrained_unishape_op")
    eis_ckpt = eis_pretrained or config.get("eis_pretrained_checkpoint") or config.get("pretrained_unishape_eis")
    if op_ckpt or eis_ckpt:
        load_weights = getattr(model, "load_official_unishape_weights", None)
        if not callable(load_weights):
            raise AttributeError("Model does not support load_official_unishape_weights")
        load_report = load_weights(op_checkpoint=op_ckpt, eis_checkpoint=eis_ckpt)
        print(f"Loaded official UniShape checkpoints: {load_report}", flush=True)
    return model


def build_optimizer(model: torch.nn.Module, experiment: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(experiment.get("lr", 1e-4)),
        weight_decay=float(experiment.get("weight_decay", 1e-4)),
    )


def _dataset_label_counts(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[int, int]:
    if not isinstance(dataset, _SizedDataset):
        return {}
    labels: list[int] = []
    for index in range(len(dataset)):
        _, _, _, label = dataset[index]
        labels.append(int(label))
    if not labels:
        return {}
    unique, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def _compute_class_weights(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    num_classes: int,
    device: torch.device,
    mode: str = "balanced",
) -> torch.Tensor | None:
    """Compute inverse-frequency class weights from the training split."""
    normalized_mode = str(mode).lower()
    if normalized_mode in {"", "none", "false", "off", "disabled"}:
        return None
    if normalized_mode not in {"balanced", "inverse_frequency"}:
        raise ValueError(f"Unsupported class_weighting mode: {mode}")
    counts_map = _dataset_label_counts(dataset)
    counts = torch.tensor(
        [float(counts_map.get(class_id, 0)) for class_id in range(int(num_classes))],
        dtype=torch.float32,
        device=device,
    )
    present = counts > 0
    if not bool(present.any()):
        return None
    weights = torch.zeros_like(counts)
    total = counts[present].sum()
    weights[present] = total / (float(present.sum().item()) * counts[present].clamp_min(1.0))
    weights[present] = weights[present] / weights[present].mean().clamp_min(1e-6)
    return weights


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_classes: int | None = None,
) -> dict[str, Any]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.no_grad():
        for x_op, x_eis, x_cond, labels in loader:
            logits, _ = model(x_op.to(device), x_eis.to(device), x_cond.to(device))
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())
    elapsed = time.perf_counter() - start
    logits_tensor = torch.cat(logits_all, dim=0)
    labels_tensor = torch.cat(labels_all, dim=0)
    preds = logits_tensor.argmax(dim=1).numpy()
    labels_np = labels_tensor.numpy()
    if num_classes is None:
        max_label = int(max(labels_np.max(initial=0), preds.max(initial=0)))
        label_ids = list(range(max_label + 1))
    else:
        label_ids = list(range(int(num_classes)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        report = cast(
            dict[str, Any],
            classification_report(labels_np, preds, labels=label_ids, output_dict=True, zero_division="warn"),
        )
    per_class_f1 = [
        {
            "class_id": class_id,
            "precision": float(report[str(class_id)]["precision"]),
            "recall": float(report[str(class_id)]["recall"]),
            "f1": float(report[str(class_id)]["f1-score"]),
            "support": int(report[str(class_id)]["support"]),
        }
        for class_id in label_ids
    ]
    return {
        "accuracy": float(accuracy_score(labels_np, preds)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion_matrix(labels_np, preds, labels=label_ids).tolist(),
        "inference_time_s": float(elapsed),
        "inference_time_per_sample_ms": float(elapsed * 1000.0 / max(len(labels_np), 1)),
        "predictions": preds.tolist(),
        "labels": labels_np.tolist(),
    }


def _write_confusion_matrix(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(metrics["confusion_matrix"])


def _write_normalized_confusion_matrix(path: Path, metrics: dict[str, Any]) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    row_sum = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    normalized = matrix / row_sum
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(normalized.tolist())


def _write_predictions(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "label", "prediction"])
        for index, (label, pred) in enumerate(zip(metrics["labels"], metrics["predictions"])):
            writer.writerow([index, label, pred])


def _write_classification_report(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(metrics.get("per_class_f1", []))


def save_outputs(output_dir: Path, metrics: dict[str, Any], param_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_save = dict(metrics)
    metrics_to_save["parameter_count"] = int(param_count)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_to_save, handle, indent=2, ensure_ascii=False)
    _write_confusion_matrix(output_dir / "val_confusion_matrix.csv", metrics)
    _write_normalized_confusion_matrix(output_dir / "val_confusion_matrix_normalized.csv", metrics)
    _write_predictions(output_dir / "val_predictions.csv", metrics)
    _write_classification_report(output_dir / "val_classification_report.csv", metrics)
    _write_confusion_matrix(output_dir / "confusion_matrix.csv", metrics)
    _write_normalized_confusion_matrix(output_dir / "confusion_matrix_normalized.csv", metrics)
    _write_predictions(output_dir / "predictions.csv", metrics)
    _write_classification_report(output_dir / "classification_report.csv", metrics)
    if "test" in metrics and isinstance(metrics["test"], dict):
        _write_confusion_matrix(output_dir / "test_confusion_matrix.csv", metrics["test"])
        _write_normalized_confusion_matrix(output_dir / "test_confusion_matrix_normalized.csv", metrics["test"])
        _write_predictions(output_dir / "test_predictions.csv", metrics["test"])
        _write_classification_report(output_dir / "test_classification_report.csv", metrics["test"])
    test_metrics = metrics.get("test") if isinstance(metrics.get("test"), dict) else None
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "val_accuracy",
            "val_macro_f1",
            "val_weighted_f1",
            "val_inference_ms",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            "test_inference_ms",
            "parameter_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "val_accuracy": metrics_to_save["accuracy"],
            "val_macro_f1": metrics_to_save["macro_f1"],
            "val_weighted_f1": metrics_to_save["weighted_f1"],
            "val_inference_ms": metrics_to_save["inference_time_per_sample_ms"],
            "test_accuracy": test_metrics.get("accuracy", "") if test_metrics else "",
            "test_macro_f1": test_metrics.get("macro_f1", "") if test_metrics else "",
            "test_weighted_f1": test_metrics.get("weighted_f1", "") if test_metrics else "",
            "test_inference_ms": test_metrics.get("inference_time_per_sample_ms", "") if test_metrics else "",
            "parameter_count": param_count,
        })


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> float:
    model.train()
    total = 0.0
    seen = 0
    for x_op, x_eis, x_cond, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, loss_dict = model(
            x_op.to(device),
            x_eis.to(device),
            x_cond.to(device),
            labels.to(device),
            class_weights=class_weights,
        )
        del logits
        loss = loss_dict["total_loss"]
        if loss is None:
            raise RuntimeError("Model did not return total_loss while labels were provided")
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        seen += batch_size
    return total / max(seen, 1)


def run_training(
    config: dict[str, Any],
    data_path: str | Path,
    output_dir: str | Path | None = None,
    op_pretrained: str | None = None,
    eis_pretrained: str | None = None,
    epochs_override: int | None = None,
    patience_override: int | None = None,
    min_delta_override: float | None = None,
    batch_size_override: int | None = None,
    refit_trainval_override: bool | None = None,
    min_epochs_before_stop_override: int | None = None,
    val_metric_smoothing_override: int | None = None,
    class_weighting_override: str | None = None,
) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    seed = int((experiment.get("seeds") or [42])[0])
    set_reproducible_seed(seed)
    print("=" * 72, flush=True)
    print("开始训练 Official CAPT-UniShape", flush=True)
    print(f"数据文件: {data_path}", flush=True)
    print(f"随机种子: {seed}", flush=True)
    train_ds, val_ds, test_ds = build_datasets_from_npz(data_path, seed=seed, val_ratio=float(experiment.get("val_ratio", 0.2)))
    probe_dataset = train_ds if isinstance(train_ds, FuelCellNPZDataset) else FuelCellNPZDataset(data_path)
    config = sync_config_with_dataset(config, probe_dataset)
    batch_size = int(batch_size_override or experiment.get("batch_size", 8))
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dir = Path(output_dir or experiment.get("output_dir", "results/official_unishape_run"))
    max_epochs = int(epochs_override or experiment.get("epochs", 5))
    patience = int(patience_override if patience_override is not None else experiment.get("patience", 10))
    min_delta = float(min_delta_override if min_delta_override is not None else experiment.get("min_delta", 1e-4))
    refit_trainval = bool(refit_trainval_override if refit_trainval_override is not None else experiment.get("refit_trainval", False))
    min_epochs_before_stop = int(
        min_epochs_before_stop_override
        if min_epochs_before_stop_override is not None
        else experiment.get("min_epochs_before_stop", 0)
    )
    val_metric_smoothing = max(
        1,
        int(
            val_metric_smoothing_override
            if val_metric_smoothing_override is not None
            else experiment.get("val_metric_smoothing", 1)
        ),
    )
    print(f"设备: {device}", flush=True)
    if not isinstance(train_ds, _SizedDataset) or not isinstance(val_ds, _SizedDataset):
        raise TypeError("train_ds and val_ds must provide __len__")
    n_train = len(train_ds)
    n_val = len(val_ds)
    n_test = len(test_ds) if test_ds is not None else 0
    print(f"训练/验证/测试样本数: {n_train} / {n_val} / {n_test}", flush=True)
    print(f"类别分布 train: {_dataset_label_counts(train_ds)}", flush=True)
    print(f"类别分布 val:   {_dataset_label_counts(val_ds)}", flush=True)
    if test_ds is not None:
        print(f"类别分布 test:  {_dataset_label_counts(test_ds)}", flush=True)
    print(f"batch_size: {batch_size}, epochs: {max_epochs}", flush=True)
    print(
        "早停: "
        f"monitor=val_macro_f1, patience={patience}, min_delta={min_delta}, "
        f"min_epochs_before_stop={min_epochs_before_stop}, smoothing={val_metric_smoothing}",
        flush=True,
    )
    if refit_trainval:
        print("最终模型策略: 先用验证集选择 epoch，再用 train+val 重新训练最终模型。", flush=True)
    print(
        "输入形状: "
        f"x_op=[C={config['c_op']}, T={config['op_seq_len']}], "
        f"x_eis=[C={config['c_eis']}, F={config['eis_seq_len']}], "
        f"x_cond=[D={config['d_cond']}], classes={config['num_classes']}",
        flush=True,
    )
    print(f"输出目录: {target_dir}", flush=True)
    model = build_experiment_model(config, device, op_pretrained=op_pretrained, eis_pretrained=eis_pretrained)
    optimizer = build_optimizer(model, experiment)
    class_weighting = str(class_weighting_override if class_weighting_override is not None else experiment.get("class_weighting", "balanced"))
    class_weights = _compute_class_weights(train_ds, int(config["num_classes"]), device, mode=class_weighting)
    if class_weights is None:
        print("类别加权 CE: disabled", flush=True)
    else:
        print(f"类别加权 CE: mode={class_weighting}, weights={class_weights.detach().cpu().tolist()}", flush=True)
    best_selection_score = -1.0
    best_raw_macro_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights=class_weights)
        metrics = evaluate_loader(model, val_loader, device, num_classes=int(config["num_classes"]))
        recent_macro_f1 = [row["val_macro_f1"] for row in history[-(val_metric_smoothing - 1):]] if val_metric_smoothing > 1 else []
        recent_macro_f1.append(float(metrics["macro_f1"]))
        selection_score = float(np.mean(np.asarray(recent_macro_f1, dtype=np.float64)))
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_macro_f1": float(metrics["macro_f1"]),
                "selection_score": selection_score,
            }
        )
        print(
            f"Epoch {epoch + 1:03d}/{max_epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_acc={metrics['accuracy']:.4f} | "
            f"val_macro_f1={metrics['macro_f1']:.4f} | "
            f"selection_score={selection_score:.4f} | "
            f"infer_ms={metrics['inference_time_per_sample_ms']:.2f}",
            flush=True,
        )
        improved = selection_score > best_selection_score + min_delta
        if improved:
            best_selection_score = selection_score
            best_raw_macro_f1 = float(metrics["macro_f1"])
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            print(
                f"  保存当前最佳模型: val_macro_f1={best_raw_macro_f1:.4f}, "
                f"selection_score={best_selection_score:.4f}",
                flush=True,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  早停计数: {epochs_without_improvement}/{patience}"
                f"（最佳 epoch={best_epoch}, best_val_macro_f1={best_raw_macro_f1:.4f}, "
                f"best_selection_score={best_selection_score:.4f}）",
                flush=True,
            )
            can_stop = (epoch + 1) >= min_epochs_before_stop
            if patience > 0 and can_stop and epochs_without_improvement >= patience:
                print(f"触发早停：验证集 macro-F1 连续 {patience} 轮未提升。", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    selection_val_metrics = evaluate_loader(model, val_loader, device, num_classes=int(config["num_classes"]))
    final_model = model
    model_selection_rule = "best_val_checkpoint"
    if refit_trainval:
        selected_epochs = max(1, int(best_epoch or len(history) or max_epochs))
        print(
            f"使用 train+val 重新训练最终模型 {selected_epochs} 轮；验证集仅用于选择 epoch，不作为最终独立验证集。",
            flush=True,
        )
        set_reproducible_seed(seed)
        trainval_ds = ConcatDataset([train_ds, val_ds])
        trainval_loader = make_loader(trainval_ds, batch_size=batch_size, shuffle=True, seed=seed + 100_003)
        final_model = build_experiment_model(config, device, op_pretrained=op_pretrained, eis_pretrained=eis_pretrained)
        final_optimizer = build_optimizer(final_model, experiment)
        final_class_weights = _compute_class_weights(trainval_ds, int(config["num_classes"]), device, mode=class_weighting)
        for refit_epoch in range(selected_epochs):
            refit_loss = train_one_epoch(final_model, trainval_loader, final_optimizer, device, class_weights=final_class_weights)
            print(
                f"Refit {refit_epoch + 1:03d}/{selected_epochs:03d} | trainval_loss={refit_loss:.6f}",
                flush=True,
            )
        model_selection_rule = "best_val_epoch_then_refit_trainval"
    final_metrics = dict(selection_val_metrics)
    final_metrics["selection_val"] = selection_val_metrics
    final_metrics["model_selection_rule"] = model_selection_rule
    final_metrics["refit_trainval"] = bool(refit_trainval)
    final_metrics["class_weighting"] = class_weighting
    final_metrics["class_weights"] = class_weights.detach().cpu().tolist() if class_weights is not None else None
    if refit_trainval:
        final_metrics["refit_val_in_sample"] = evaluate_loader(final_model, val_loader, device, num_classes=int(config["num_classes"]))
    if test_ds is not None:
        test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False, seed=seed)
        final_metrics["test"] = evaluate_loader(final_model, test_loader, device, num_classes=int(config["num_classes"]))
        test_metrics = final_metrics["test"]
        print(
            f"测试集结果: acc={test_metrics['accuracy']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"infer_ms={test_metrics['inference_time_per_sample_ms']:.2f}",
            flush=True,
        )
    param_count = count_parameters(final_model)
    save_outputs(target_dir, final_metrics, param_count)
    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "config": config,
                "history": history,
                "early_stopping": {
                    "monitor": "val_macro_f1",
                    "selection_score": "smoothed_val_macro_f1",
                    "patience": patience,
                    "min_delta": min_delta,
                    "min_epochs_before_stop": min_epochs_before_stop,
                    "val_metric_smoothing": val_metric_smoothing,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_raw_macro_f1,
                    "best_selection_score": best_selection_score,
                    "epochs_ran": len(history),
                    "class_weighting": class_weighting,
                },
            },
            target_dir / "best_val.ckpt",
        )
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "config": config,
            "history": history,
            "early_stopping": {
                "monitor": "val_macro_f1",
                "selection_score": "smoothed_val_macro_f1",
                "patience": patience,
                "min_delta": min_delta,
                "min_epochs_before_stop": min_epochs_before_stop,
                "val_metric_smoothing": val_metric_smoothing,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_raw_macro_f1,
                "best_selection_score": best_selection_score,
                "epochs_ran": len(history),
                "model_selection_rule": model_selection_rule,
                "refit_trainval": bool(refit_trainval),
                "class_weighting": class_weighting,
            },
        },
        target_dir / "best.ckpt",
    )
    print("训练完成。已保存:", flush=True)
    print(f"  {target_dir / 'best.ckpt'}", flush=True)
    print(f"  {target_dir / 'metrics.json'}", flush=True)
    print(f"  {target_dir / 'summary.csv'}", flush=True)
    print(f"  {target_dir / 'confusion_matrix.csv'}", flush=True)
    print(f"  {target_dir / 'predictions.csv'}", flush=True)
    print(f"  {target_dir / 'val_classification_report.csv'}", flush=True)
    if test_ds is not None:
        print(f"  {target_dir / 'test_classification_report.csv'}", flush=True)
    print("=" * 72, flush=True)
    return final_metrics


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train official CAPT-UniShape RBF/No-RBF models")
    parser.add_argument("--config", default="configs/rbf_kanfusion.yaml")
    parser.add_argument("--data", required=True, help="NPZ file with x_op, x_eis, x_cond/cond and labels/y")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--op-pretrained", default=None, help="Official UniShape checkpoint for operation branch")
    parser.add_argument("--eis-pretrained", default=None, help="Official UniShape checkpoint for EIS branch")
    parser.add_argument("--epochs", type=int, default=None, help="Temporarily override experiment.epochs")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience; <=0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=None, help="Minimum val_macro_f1 improvement required by early stopping")
    parser.add_argument("--batch-size", type=int, default=None, help="Temporarily override experiment.batch_size")
    parser.add_argument("--refit-trainval", action=argparse.BooleanOptionalAction, default=None, help="Select epoch on val, then refit final model on train+val")
    parser.add_argument("--min-epochs-before-stop", type=int, default=None, help="Do not trigger early stopping before this epoch")
    parser.add_argument("--val-metric-smoothing", type=int, default=None, help="Moving average window for val macro-F1 selection score")
    parser.add_argument("--class-weighting", choices=["balanced", "inverse_frequency", "none"], default=None, help="Class weighting mode for neural-network CE loss")
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    run_training(
        load_config(args.config),
        args.data,
        args.output_dir,
        args.op_pretrained,
        args.eis_pretrained,
        args.epochs,
        args.patience,
        args.min_delta,
        args.batch_size,
        args.refit_trainval,
        args.min_epochs_before_stop,
        args.val_metric_smoothing,
        args.class_weighting,
    )


if __name__ == "__main__":
    main()
