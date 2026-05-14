"""Run proposed CAPT-UniShape against ML/DL/Transformer/iTransformer baselines."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_converter = importlib.import_module("experiments.build_official_npz_from_self_excel")
_train = importlib.import_module("train")
build_npz = _converter.build_npz
load_config = _train.load_config
run_training = _train.run_training


RATIO_TO_TEST_SIZE = {
    "8_2": 0.20,
    "7_3": 0.30,
    "6_4": 0.40,
    "5_5": 0.50,
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


def _ratio_train_fraction(ratio: str, fixed_test_ratio: str) -> float:
    """Map a nominal train/test ratio to a train-subset fraction under fixed test."""
    base_train_pool = 1.0 - float(RATIO_TO_TEST_SIZE[fixed_test_ratio])
    target_train_pool = 1.0 - float(RATIO_TO_TEST_SIZE[ratio])
    if base_train_pool <= 0:
        raise ValueError("fixed_test_ratio leaves no training pool")
    fraction = target_train_pool / base_train_pool
    if fraction > 1.0 + 1e-12:
        raise ValueError(f"ratio={ratio} needs more training data than fixed_test_ratio={fixed_test_ratio}")
    return min(1.0, max(0.0, fraction))


def _label_count_dict(values: np.ndarray[Any, Any]) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def _make_train_subset_npz(
    base_npz: Path,
    output_npz: Path,
    train_fraction: float,
    seed: int,
    split_protocol: str = "fixed_test_train_subset",
) -> Path:
    """Keep val/test fixed and stratified-subsample only train windows."""
    data = np.load(base_npz)
    if "split" not in data:
        raise ValueError(f"{base_npz} 缺少 split 数组，无法固定验证/测试集")
    label_key = "labels" if "labels" in data else "y"
    labels = np.asarray(data[label_key], dtype=np.int64)
    split = np.asarray(data["split"], dtype=np.int64)
    train_indices = np.where(split == 0)[0]
    heldout_indices = np.where(split != 0)[0]
    rng = np.random.default_rng(int(seed))
    selected_train: list[int] = []
    for label in sorted(np.unique(labels[train_indices]).tolist()):
        class_indices = train_indices[labels[train_indices] == label].copy()
        rng.shuffle(class_indices)
        keep = int(round(len(class_indices) * float(train_fraction)))
        keep = max(1, min(len(class_indices), keep))
        selected_train.extend(class_indices[:keep].tolist())
    selected = np.asarray(sorted(selected_train) + heldout_indices.tolist(), dtype=np.int64)
    payload: dict[str, np.ndarray[Any, Any]] = {}
    total_samples = int(split.shape[0])
    for key in data.files:
        array = np.asarray(data[key])
        if array.ndim > 0 and int(array.shape[0]) == total_samples:
            payload[key] = array[selected].copy()
        else:
            payload[key] = array.copy()
    payload["split"] = split[selected].copy()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    split_label_counts = {
        "train": _label_count_dict(labels[selected][payload["split"] == 0]),
        "val": _label_count_dict(labels[selected][payload["split"] == 1]),
        "test": _label_count_dict(labels[selected][payload["split"] == 2]),
    }
    summary: dict[str, Any] = {
        "source_npz": str(base_npz),
        "output_path": str(output_npz),
        "split_protocol": split_protocol,
        "train_fraction_from_fixed_pool": float(train_fraction),
        "num_samples": int(len(selected)),
        "train_size": int((payload["split"] == 0).sum()),
        "val_size": int((payload["split"] == 1).sum()),
        "test_size": int((payload["split"] == 2).sum()),
        "split_label_counts": split_label_counts,
    }
    original_train_counts = _label_count_dict(labels[train_indices])
    subset_train_counts = split_label_counts["train"]
    summary["train_subset_diagnostics"] = {
        "original_train_counts": original_train_counts,
        "subset_train_counts": subset_train_counts,
        "per_class_retained_fraction": {
            label: float(subset_train_counts.get(label, 0) / max(count, 1))
            for label, count in original_train_counts.items()
        },
        "min_train_class_support": min(subset_train_counts.values()) if subset_train_counts else 0,
    }
    output_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_npz


def _make_nested_train_subset_npzs(
    base_npz: Path,
    ratio_to_output_npz: dict[str, Path],
    fixed_test_ratio: str,
    seed: int,
) -> dict[str, Path]:
    """Create fixed-test train subsets with one shared class order.

    Using the same class-wise shuffle for every ratio guarantees that smaller
    training pools are strict subsets of larger ones, e.g. 5:5 <= 6:4 <= 7:3.
    """
    outputs: dict[str, Path] = {}
    for ratio, output_npz in ratio_to_output_npz.items():
        train_fraction = _ratio_train_fraction(ratio, fixed_test_ratio)
        outputs[ratio] = _make_train_subset_npz(
            base_npz,
            output_npz,
            train_fraction=train_fraction,
            seed=seed,
            split_protocol="nested_fixed_test_train_subset",
        )
    return outputs


class BaselineNPZDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Split-aware NPZ dataset for baseline models."""

    def __init__(self, npz_path: Path, split_value: int) -> None:
        data = np.load(npz_path)
        if "split" not in data:
            raise ValueError(f"{npz_path} 缺少 split 数组")
        split = np.asarray(data["split"], dtype=np.int64)
        indices = np.where(split == int(split_value))[0]
        if len(indices) == 0:
            raise ValueError(f"{npz_path} 没有 split={split_value} 的样本")
        label_key = "labels" if "labels" in data else "y"
        cond_key = "x_cond" if "x_cond" in data else "cond"
        self.x_op = torch.as_tensor(data["x_op"][indices], dtype=torch.float32)
        self.x_eis = torch.as_tensor(data["x_eis"][indices], dtype=torch.float32)
        self.x_cond = torch.as_tensor(data[cond_key][indices], dtype=torch.float32)
        self.labels = torch.as_tensor(data[label_key][indices], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_op[index], self.x_eis[index], self.x_cond[index], self.labels[index]


def _combined_sequence(x_op: torch.Tensor, x_eis: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
    target_len = max(int(x_op.shape[-1]), int(x_eis.shape[-1]))
    op = F.interpolate(x_op, size=target_len, mode="linear", align_corners=False) if x_op.shape[-1] != target_len else x_op
    eis = F.interpolate(x_eis, size=target_len, mode="linear", align_corners=False) if x_eis.shape[-1] != target_len else x_eis
    cond = x_cond.unsqueeze(-1).expand(-1, -1, target_len)
    return torch.cat([op, eis, cond], dim=1)


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_op: torch.Tensor, x_eis: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x_op.flatten(1), x_eis.flatten(1), x_cond], dim=1)
        return self.net(features)


class CNN1DBaseline(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_op: torch.Tensor, x_eis: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        return self.net(_combined_sequence(x_op, x_eis, x_cond))


class LSTMBaseline(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))

    def forward(self, x_op: torch.Tensor, x_eis: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        sequence = _combined_sequence(x_op, x_eis, x_cond).transpose(1, 2)
        output, _ = self.lstm(sequence)
        return self.head(output.mean(dim=1))


class TransformerBaseline(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, d_model: int, num_classes: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x_op: torch.Tensor, x_eis: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        sequence = _combined_sequence(x_op, x_eis, x_cond).transpose(1, 2)
        encoded = self.encoder(self.input_proj(sequence) + self.pos_embed[:, : sequence.shape[1]])
        return self.head(encoded.mean(dim=1))


class ITransformerBaseline(nn.Module):
    """iTransformer-style classifier: variables are tokens, time is embedded."""

    def __init__(
        self,
        c_op: int,
        op_len: int,
        c_eis: int,
        eis_len: int,
        d_cond: int,
        d_model: int,
        num_classes: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.op_proj = nn.Linear(op_len, d_model)
        self.eis_proj = nn.Linear(eis_len, d_model)
        self.op_variable_embed = nn.Parameter(torch.zeros(1, c_op, d_model))
        self.eis_variable_embed = nn.Parameter(torch.zeros(1, c_eis, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.op_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        eis_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.eis_encoder = nn.TransformerEncoder(eis_layer, num_layers=num_layers)
        self.cond_encoder = nn.Sequential(nn.Linear(d_cond, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, num_classes))

    def forward(self, x_op: torch.Tensor, x_eis: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        op_tokens = self.op_encoder(self.op_proj(x_op) + self.op_variable_embed[:, : x_op.shape[1]])
        eis_tokens = self.eis_encoder(self.eis_proj(x_eis) + self.eis_variable_embed[:, : x_eis.shape[1]])
        z_cond = self.cond_encoder(x_cond)
        return self.head(torch.cat([op_tokens.mean(dim=1), eis_tokens.mean(dim=1), z_cond], dim=1))


def _num_classes(npz_path: Path) -> int:
    data = np.load(npz_path)
    label_key = "labels" if "labels" in data else "y"
    return int(np.max(data[label_key])) + 1


def _flatten_split(npz_path: Path, split_value: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    data = np.load(npz_path)
    split = np.asarray(data["split"], dtype=np.int64)
    indices = np.where(split == int(split_value))[0]
    cond_key = "x_cond" if "x_cond" in data else "cond"
    label_key = "labels" if "labels" in data else "y"
    features = np.concatenate(
        [
            np.asarray(data["x_op"][indices], dtype=np.float32).reshape(len(indices), -1),
            np.asarray(data["x_eis"][indices], dtype=np.float32).reshape(len(indices), -1),
            np.asarray(data[cond_key][indices], dtype=np.float32).reshape(len(indices), -1),
        ],
        axis=1,
    )
    labels = np.asarray(data[label_key][indices], dtype=np.int64)
    return features, labels


def _classification_metrics(
    labels: np.ndarray[Any, Any],
    preds: np.ndarray[Any, Any],
    elapsed: float,
    num_classes: int,
) -> dict[str, Any]:
    label_ids = list(range(int(num_classes)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        report = cast(
            dict[str, Any],
            classification_report(labels, preds, labels=label_ids, output_dict=True, zero_division="warn"),
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
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion_matrix(labels, preds, labels=label_ids).tolist(),
        "inference_time_s": float(elapsed),
        "inference_time_per_sample_ms": float(elapsed * 1000.0 / max(len(labels), 1)),
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
}


def _write_classification_report(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(metrics.get("per_class_f1", []))


def _write_normalized_confusion_matrix(path: Path, metrics: dict[str, Any]) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    row_sum = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    normalized = matrix / row_sum
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(normalized.tolist())


def _save_result(
    output_dir: Path,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    param_count: int,
    extra_payload: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(val_metrics)
    payload["test"] = test_metrics
    payload["parameter_count"] = int(param_count)
    payload["artifact_semantics"] = {
        "top_level_metrics": "selection_validation",
        "non_prefixed_confusion_matrix_csv": "test_backward_compatibility",
        "test_prefixed_files": "authoritative_test_artifacts",
        "val_prefixed_files": "selection_validation_artifacts",
    }
    if extra_payload:
        payload.update(extra_payload)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "val_accuracy",
                "val_macro_f1",
                "val_weighted_f1",
                "val_inference_ms",
                "test_accuracy",
                "test_macro_f1",
                "test_weighted_f1",
                "test_inference_ms",
                "parameter_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_inference_ms": val_metrics["inference_time_per_sample_ms"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_inference_ms": test_metrics["inference_time_per_sample_ms"],
                "parameter_count": int(param_count),
            }
        )
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(test_metrics["confusion_matrix"])
    _write_normalized_confusion_matrix(output_dir / "confusion_matrix_normalized.csv", test_metrics)
    with (output_dir / "test_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(test_metrics["confusion_matrix"])
    _write_normalized_confusion_matrix(output_dir / "test_confusion_matrix_normalized.csv", test_metrics)
    with (output_dir / "val_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(val_metrics["confusion_matrix"])
    _write_normalized_confusion_matrix(output_dir / "val_confusion_matrix_normalized.csv", val_metrics)
    _write_classification_report(output_dir / "classification_report.csv", test_metrics)
    _write_classification_report(output_dir / "test_classification_report.csv", test_metrics)
    _write_classification_report(output_dir / "val_classification_report.csv", val_metrics)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "label", "prediction"])
        for index, (label, pred) in enumerate(zip(test_metrics["labels"], test_metrics["predictions"])):
            writer.writerow([index, label, pred])
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "label", "prediction"])
        for index, (label, pred) in enumerate(zip(test_metrics["labels"], test_metrics["predictions"])):
            writer.writerow([index, label, pred])
    with (output_dir / "val_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "label", "prediction"])
        for index, (label, pred) in enumerate(zip(val_metrics["labels"], val_metrics["predictions"])):
            writer.writerow([index, label, pred])
    return metrics_path


def _metric_row(ratio: str, model_key: str, metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_payload = payload.get("test", payload)
    test_source = "test" if "test" in payload else "top_level_fallback"
    return {
        "ratio": ratio.replace("_", ":"),
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


def _print_metric_row(row: dict[str, Any]) -> None:
    print(
        "测试集效果 | "
        f"ratio={row['ratio']} | model={row['model']} | "
        f"category={row['category']} | "
        f"test_acc={float(row['test_accuracy']):.4f} | "
        f"test_macro_f1={float(row['test_macro_f1']):.4f} | "
        f"infer_ms={float(row['test_inference_ms']):.2f}",
        flush=True,
    )


def _ml_parameter_count(model: Any) -> int:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    total = 0
    for attr in ("coef_", "intercept_", "feature_importances_"):
        value = getattr(estimator, attr, None)
        if value is not None:
            total += int(np.asarray(value).size)
    for attr in ("support_vectors_", "dual_coef_"):
        value = getattr(estimator, attr, None)
        if value is not None:
            total += int(np.asarray(value).size)
    estimators = getattr(estimator, "estimators_", None)
    if estimators is not None:
        total += int(sum(getattr(tree, "tree_", tree).node_count for tree in estimators))
    return total


def _torch_class_weights(labels: torch.Tensor, num_classes: int, device: torch.device, mode: str) -> torch.Tensor | None:
    normalized_mode = str(mode).lower()
    if normalized_mode in {"", "none", "false", "off", "disabled"}:
        return None
    if normalized_mode not in {"sqrt_balanced", "balanced", "inverse_frequency", "effective_number"}:
        raise ValueError(f"Unsupported class_weighting mode: {mode}")
    counts = torch.bincount(labels, minlength=num_classes).to(device=device, dtype=torch.float32)
    present = counts > 0
    if not bool(present.any()):
        return None
    weights = torch.zeros_like(counts)
    if normalized_mode == "effective_number":
        beta = 0.999
        weights[present] = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts[present], beta), counts[present].clamp_min(1.0)))
    else:
        weights[present] = counts[present].sum() / (float(present.sum().item()) * counts[present].clamp_min(1.0))
        if normalized_mode == "sqrt_balanced":
            weights[present] = torch.sqrt(weights[present])
    weights[present] = weights[present] / weights[present].mean().clamp_min(1e-6)
    return weights


def _build_ml_model(model_key: str, seed: int, rf_estimators: int) -> Any:
    if model_key == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
    if model_key == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"))
    if model_key == "random_forest":
        return RandomForestClassifier(n_estimators=rf_estimators, class_weight="balanced", random_state=seed, n_jobs=-1)
    raise KeyError(model_key)


def run_ml_baseline(model_key: str, npz_path: Path, output_dir: Path, seed: int, rf_estimators: int, refit_trainval: bool) -> Path:
    num_classes = _num_classes(npz_path)
    x_train, y_train = _flatten_split(npz_path, split_value=0)
    x_val, y_val = _flatten_split(npz_path, split_value=1)
    x_test, y_test = _flatten_split(npz_path, split_value=2)
    model = _build_ml_model(model_key, seed=seed, rf_estimators=rf_estimators)
    model.fit(x_train, y_train)
    start = time.perf_counter()
    selection_val_preds = model.predict(x_val)
    selection_val_elapsed = time.perf_counter() - start
    selection_val_metrics = _classification_metrics(y_val, selection_val_preds, selection_val_elapsed, num_classes)
    if refit_trainval:
        x_fit = np.concatenate([x_train, x_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        model = _build_ml_model(model_key, seed=seed, rf_estimators=rf_estimators)
        model.fit(x_fit, y_fit)

    start = time.perf_counter()
    refit_val_preds = model.predict(x_val)
    refit_val_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    test_preds = model.predict(x_test)
    test_elapsed = time.perf_counter() - start
    refit_val_metrics = _classification_metrics(y_val, refit_val_preds, refit_val_elapsed, num_classes)
    test_metrics = _classification_metrics(y_test, test_preds, test_elapsed, num_classes)
    extra_payload = {
        "selection_val": selection_val_metrics,
        "refit_trainval": bool(refit_trainval),
        "model_selection_rule": "train_then_refit_trainval" if refit_trainval else "train_only",
    }
    if refit_trainval:
        extra_payload["refit_val_in_sample"] = refit_val_metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    extra_payload["model_artifact_path"] = str(model_path)
    return _save_result(output_dir, selection_val_metrics, test_metrics, _ml_parameter_count(model), extra_payload=extra_payload)


def _evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    preds_all: list[np.ndarray[Any, Any]] = []
    labels_all: list[np.ndarray[Any, Any]] = []
    start = time.perf_counter()
    with torch.no_grad():
        for x_op, x_eis, x_cond, labels in loader:
            logits = model(x_op.to(device), x_eis.to(device), x_cond.to(device))
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    elapsed = time.perf_counter() - start
    preds = np.concatenate(preds_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    return _classification_metrics(labels, preds, elapsed, num_classes)


def _build_torch_model(
    model_key: str,
    train_ds: BaselineNPZDataset,
    num_classes: int,
    hidden_dim: int,
    d_model: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    c_op = int(train_ds.x_op.shape[1])
    op_len = int(train_ds.x_op.shape[2])
    c_eis = int(train_ds.x_eis.shape[1])
    eis_len = int(train_ds.x_eis.shape[2])
    d_cond = int(train_ds.x_cond.shape[1])
    in_channels = c_op + c_eis + d_cond
    seq_len = max(op_len, eis_len)
    if model_key == "mlp":
        return MLPBaseline(c_op * op_len + c_eis * eis_len + d_cond, hidden_dim, num_classes, dropout)
    if model_key == "cnn1d":
        return CNN1DBaseline(in_channels, hidden_dim, num_classes, dropout)
    if model_key == "lstm":
        return LSTMBaseline(in_channels, hidden_dim, num_classes, dropout)
    if model_key == "transformer":
        return TransformerBaseline(in_channels, seq_len, d_model, num_classes, num_layers, dropout)
    if model_key == "itransformer":
        return ITransformerBaseline(c_op, op_len, c_eis, eis_len, d_cond, d_model, num_classes, num_layers, dropout)
    raise KeyError(model_key)


def run_torch_baseline(
    model_key: str,
    npz_path: Path,
    output_dir: Path,
    epochs: int,
    patience: int,
    min_delta: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    hidden_dim: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    refit_trainval: bool,
    min_epochs_before_stop: int,
    val_metric_smoothing: int,
    class_weighting: str,
) -> Path:
    torch.manual_seed(seed)
    num_classes = _num_classes(npz_path)
    train_ds = BaselineNPZDataset(npz_path, split_value=0)
    val_ds = BaselineNPZDataset(npz_path, split_value=1)
    test_ds = BaselineNPZDataset(npz_path, split_value=2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_torch_model(model_key, train_ds, num_classes, hidden_dim, d_model, num_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = _torch_class_weights(train_ds.labels, num_classes, device, class_weighting)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if class_weights is None:
        print(f"{model_key}: 类别加权 CE disabled", flush=True)
    else:
        print(f"{model_key}: 类别加权 CE mode={class_weighting}, weights={class_weights.detach().cpu().tolist()}", flush=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_selection_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    val_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for x_op, x_eis, x_cond, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_op.to(device), x_eis.to(device), x_cond.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            batch_size_seen = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size_seen
            seen += batch_size_seen
        val_metrics = _evaluate_torch_model(model, val_loader, device, num_classes)
        val_history.append(float(val_metrics["macro_f1"]))
        window = max(1, int(val_metric_smoothing))
        selection_score = float(np.mean(np.asarray(val_history[-window:], dtype=np.float64)))
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | {model_key} | "
            f"train_loss={total_loss / max(seen, 1):.6f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"selection_score={selection_score:.4f}",
            flush=True,
        )
        improved = selection_score > best_selection_score + min_delta
        if improved:
            best_selection_score = selection_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if patience > 0 and (epoch + 1) >= int(min_epochs_before_stop) and epochs_without_improvement >= patience:
                print(f"触发早停：{model_key} 验证集 macro-F1 连续 {patience} 轮未提升。", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    selection_val_metrics = _evaluate_torch_model(model, val_loader, device, num_classes)
    final_class_weights = class_weights
    if refit_trainval:
        selected_epochs = max(1, int(best_epoch or len(val_history) or epochs))
        print(f"{model_key}: 用 train+val 重新训练最终模型 {selected_epochs} 轮。", flush=True)
        torch.manual_seed(seed)
        trainval_ds = ConcatDataset([train_ds, val_ds])
        trainval_loader = DataLoader(trainval_ds, batch_size=batch_size, shuffle=True)
        model = _build_torch_model(model_key, train_ds, num_classes, hidden_dim, d_model, num_layers, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        trainval_labels = torch.cat([train_ds.labels, val_ds.labels], dim=0)
        final_class_weights = _torch_class_weights(trainval_labels, num_classes, device, class_weighting)
        criterion = nn.CrossEntropyLoss(weight=final_class_weights)
        if final_class_weights is None:
            print(f"{model_key}: Refit 类别加权 CE disabled", flush=True)
        else:
            print(
                f"{model_key}: Refit 类别加权 CE mode={class_weighting}, "
                f"weights={final_class_weights.detach().cpu().tolist()}",
                flush=True,
            )
        for _ in range(selected_epochs):
            model.train()
            for x_op, x_eis, x_cond, labels in trainval_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_op.to(device), x_eis.to(device), x_cond.to(device))
                loss = criterion(logits, labels.to(device))
                loss.backward()
                optimizer.step()
    refit_val_metrics = _evaluate_torch_model(model, val_loader, device, num_classes)
    test_metrics = _evaluate_torch_model(model, test_loader, device, num_classes)
    param_count = int(sum(parameter.numel() for parameter in model.parameters()))
    extra_payload = {
        "selection_val": selection_val_metrics,
        "refit_trainval": bool(refit_trainval),
        "model_selection_rule": "best_val_epoch_then_refit_trainval" if refit_trainval else "best_val_checkpoint",
        "class_weighting": class_weighting,
        "class_weights": final_class_weights.detach().cpu().tolist() if final_class_weights is not None else None,
        "selection_class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "final_class_weights_source": "train_val" if refit_trainval else "train",
    }
    if refit_trainval:
        extra_payload["refit_val_in_sample"] = refit_val_metrics
    metrics_path = _save_result(output_dir, selection_val_metrics, test_metrics, param_count, extra_payload=extra_payload)
    torch.save({"model_state_dict": model.state_dict(), "model_key": model_key}, output_dir / "best.ckpt")
    return metrics_path


def run_proposed_model(
    npz_path: Path,
    output_dir: Path,
    config_path: Path,
    epochs: int,
    patience: int,
    min_delta: float,
    batch_size: int,
    refit_trainval: bool,
    min_epochs_before_stop: int,
    val_metric_smoothing: int,
    class_weighting: str,
    seed: int,
    lr: float,
    weight_decay: float,
    checkpoint_selection: str,
    selection_score: str,
    init_checkpoint: str | None,
) -> Path:
    config = load_config(config_path)
    experiment = config.setdefault("experiment", {})
    experiment["output_dir"] = str(output_dir)
    experiment["batch_size"] = int(batch_size)
    experiment["seeds"] = [int(seed)]
    run_training(
        config=config,
        data_path=npz_path,
        output_dir=output_dir,
        epochs_override=epochs,
        patience_override=patience,
        min_delta_override=min_delta,
        batch_size_override=batch_size,
        refit_trainval_override=refit_trainval,
        min_epochs_before_stop_override=min_epochs_before_stop,
        val_metric_smoothing_override=val_metric_smoothing,
        class_weighting_override=class_weighting,
        lr_override=lr,
        weight_decay_override=weight_decay,
        checkpoint_selection_override=checkpoint_selection,
        selection_score_override=selection_score,
        init_checkpoint_override=init_checkpoint,
    )
    return output_dir / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="运行提出模型与传统机器学习/深度学习/Transformer/iTransformer 基准对比实验")
    parser.add_argument("--excel", default="data/raw/水淹和膜干故障测试数据_补充特征汇总.xlsx")
    parser.add_argument("--ratios", nargs="+", default=list(RATIO_TO_TEST_SIZE.keys()), choices=list(RATIO_TO_TEST_SIZE.keys()))
    parser.add_argument("--models", nargs="+", default=list(MODEL_CATEGORIES.keys()), choices=list(MODEL_CATEGORIES.keys()))
    parser.add_argument("--output-root", default="results/official_baseline_comparison")
    parser.add_argument("--data-root", default="data/processed/official_baseline_comparison")
    parser.add_argument("--split-protocol", choices=["fixed_test", "independent"], default="fixed_test", help="fixed_test 固定同一验证/测试集，只改变训练子集大小；independent 为每个比例单独重划分")
    parser.add_argument("--fixed-test-ratio", choices=list(RATIO_TO_TEST_SIZE.keys()), default="8_2", help="fixed_test 协议使用的固定验证/测试集基准比例")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride-train", type=int, default=16)
    parser.add_argument("--stride-val", type=int, default=None)
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
    parser.add_argument("--min-epochs-before-stop", type=int, default=20, help="避免小验证集过早触发早停")
    parser.add_argument("--val-metric-smoothing", type=int, default=3, help="验证macro-F1移动平均窗口，用于降低选模噪声")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-aware-train-stride", action="store_true", help="训练集按类别自动调节滑窗步长：少数类更密，多数类更稀")
    parser.add_argument("--min-train-stride", type=int, default=None, help="类别感知训练步长下限；默认 stride_train//2")
    parser.add_argument("--max-train-stride", type=int, default=None, help="类别感知训练步长上限；默认 stride_train*2")
    parser.add_argument("--class-stride-power", type=float, default=1.0, help="类别样本数到训练步长的缩放幂指数")
    parser.add_argument("--class-weighting", choices=["sqrt_balanced", "balanced", "inverse_frequency", "effective_number", "balanced_softmax", "logit_adjusted", "none"], default="sqrt_balanced", help="神经网络模型 CE 类别权重/调整模式")
    parser.add_argument("--proposed-config", default="configs/rbf_kanfusion.yaml", help="提出模型使用的 YAML 配置路径，默认保持原 RBF/KANFusion 配置")
    parser.add_argument("--checkpoint-selection", choices=["best_val", "best-val", "last"], default="best_val", help="proposed 最终 checkpoint 选择策略")
    parser.add_argument("--selection-score", choices=["macro_f1", "macro_f1_class0_recall", "macro_f1_gap_penalty", "val_train_holdout_macro_f1"], default="macro_f1", help="验证集选模/早停评分")
    parser.add_argument("--init-checkpoint", default=None, help="proposed 全模型 warm-start checkpoint")
    parser.add_argument("--split-retries", type=int, default=50, help="重试分组划分并选择少数类支持更好的 split")
    parser.add_argument("--min-eval-class-windows", type=int, default=5, help="验证/测试集中任一类别窗口数低于该值时标记为不稳定")
    parser.add_argument("--min-eval-class-groups", type=int, default=1, help="验证/测试集中任一类别 group 数低于该值时标记为不稳定")
    parser.add_argument("--min-train-class-windows", type=int, default=1, help="训练集中第0类（normal）窗口数下限（用于 split 评分/过滤）")
    parser.add_argument("--min-train-class-groups", type=int, default=1, help="训练集中第0类（normal）group 数下限（用于 split 评分/过滤）")
    parser.add_argument("--min-val-class-groups", type=int, default=1, help="验证集中第0类（normal）group 数下限（用于 split 评分/过滤）")
    parser.add_argument("--min-test-class-groups", type=int, default=1, help="测试集中第0类（normal）group 数下限（用于 split 评分/过滤）")
    parser.add_argument("--prefer-balanced-train-groups", action=argparse.BooleanOptionalAction, default=False, help="split 评分中惩罚训练集类别 group 分布失衡")
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    data_root = ROOT / args.data_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    ml_models = {"logreg", "svm", "random_forest"}
    torch_models = {"mlp", "cnn1d", "lstm", "transformer", "itransformer"}
    fixed_base_npz: Path | None = None
    nested_fixed_npzs: dict[str, Path] = {}
    if args.split_protocol == "fixed_test":
        fixed_base_npz = data_root / f"official_self_stack_impedance_eis_w{args.window_size}_{args.fixed_test_ratio}_fixed_base.npz"
        print(f"\n=== 构建固定验证/测试基准数据: train/test={args.fixed_test_ratio.replace('_', ':')} ===", flush=True)
        build_npz(
            excel_path=ROOT / args.excel,
            output_path=fixed_base_npz,
            window_size=args.window_size,
            stride_train=args.stride_train,
            stride_val=args.stride_val,
            stride_eval=args.stride_eval,
            eis_seq_len=args.eis_seq_len,
            split_mode=args.split_mode,
            segment_gap_seconds=args.segment_gap_seconds,
            segment_block_seconds=args.segment_block_seconds,
            segment_label_boundary=args.segment_label_boundary,
            random_state=args.seed,
            op_source="stack",
            test_size=RATIO_TO_TEST_SIZE[args.fixed_test_ratio],
            val_size=args.val_size,
            class_aware_train_stride=args.class_aware_train_stride,
            min_train_stride=args.min_train_stride,
            max_train_stride=args.max_train_stride,
            class_stride_power=args.class_stride_power,
            group_split_strategy=args.group_split_strategy,
            split_retries=args.split_retries,
            min_eval_class_windows=args.min_eval_class_windows,
            min_eval_class_groups=args.min_eval_class_groups,
            min_train_class_windows=args.min_train_class_windows,
            min_train_class_groups=args.min_train_class_groups,
            min_val_class_groups=args.min_val_class_groups,
            min_test_class_groups=args.min_test_class_groups,
            prefer_balanced_train_groups=args.prefer_balanced_train_groups,
        )
        ratio_to_output = {
            ratio: data_root / f"official_self_stack_impedance_eis_w{args.window_size}_{ratio}.npz"
            for ratio in args.ratios
        }
        print("\n=== 构建嵌套固定测试协议数据：固定 val/test，仅按同一类别顺序截取训练池 ===", flush=True)
        for ratio in args.ratios:
            train_fraction = _ratio_train_fraction(ratio, args.fixed_test_ratio)
            print(
                f"nested_fixed_test | nominal={ratio.replace('_', ':')} | "
                f"train_fraction={train_fraction:.3f} | subset_seed={args.seed}",
                flush=True,
            )
        nested_fixed_npzs = _make_nested_train_subset_npzs(
            fixed_base_npz,
            ratio_to_output,
            fixed_test_ratio=args.fixed_test_ratio,
            seed=args.seed,
        )
    for ratio in args.ratios:
        npz_path = data_root / f"official_self_stack_impedance_eis_w{args.window_size}_{ratio}.npz"
        if args.split_protocol == "fixed_test":
            if fixed_base_npz is None:
                raise RuntimeError("fixed_base_npz was not created")
            npz_path = nested_fixed_npzs[ratio]
        else:
            print(f"\n=== 构建独立比例数据: train/test={ratio.replace('_', ':')} ===", flush=True)
            build_npz(
                excel_path=ROOT / args.excel,
                output_path=npz_path,
                window_size=args.window_size,
                stride_train=args.stride_train,
                stride_val=args.stride_val,
                stride_eval=args.stride_eval,
                eis_seq_len=args.eis_seq_len,
                split_mode=args.split_mode,
                segment_gap_seconds=args.segment_gap_seconds,
                segment_block_seconds=args.segment_block_seconds,
                segment_label_boundary=args.segment_label_boundary,
                random_state=args.seed,
                op_source="stack",
                test_size=RATIO_TO_TEST_SIZE[ratio],
                val_size=args.val_size,
                class_aware_train_stride=args.class_aware_train_stride,
                min_train_stride=args.min_train_stride,
                max_train_stride=args.max_train_stride,
                class_stride_power=args.class_stride_power,
                group_split_strategy=args.group_split_strategy,
                split_retries=args.split_retries,
                min_eval_class_windows=args.min_eval_class_windows,
                min_eval_class_groups=args.min_eval_class_groups,
                min_train_class_windows=args.min_train_class_windows,
                min_train_class_groups=args.min_train_class_groups,
                min_val_class_groups=args.min_val_class_groups,
                min_test_class_groups=args.min_test_class_groups,
                prefer_balanced_train_groups=args.prefer_balanced_train_groups,
            )
        for model_key in args.models:
            run_dir = output_root / ratio / model_key
            print(f"\n=== 对比实验: {model_key} | train/test={ratio.replace('_', ':')} ===", flush=True)
            if model_key == "proposed":
                metrics_path = run_proposed_model(
                    npz_path,
                    run_dir,
                    ROOT / args.proposed_config,
                    args.epochs,
                    args.patience,
                    args.min_delta,
                    args.batch_size,
                    args.refit_trainval,
                    args.min_epochs_before_stop,
                    args.val_metric_smoothing,
                    args.class_weighting,
                    args.seed,
                    args.lr,
                    args.weight_decay,
                    args.checkpoint_selection,
                    args.selection_score,
                    args.init_checkpoint,
                )
            elif model_key in ml_models:
                metrics_path = run_ml_baseline(model_key, npz_path, run_dir, args.seed, args.rf_estimators, args.refit_trainval)
            elif model_key in torch_models:
                metrics_path = run_torch_baseline(
                    model_key=model_key,
                    npz_path=npz_path,
                    output_dir=run_dir,
                    epochs=args.epochs,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    seed=args.seed,
                    hidden_dim=args.hidden_dim,
                    d_model=args.d_model,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    refit_trainval=args.refit_trainval,
                    min_epochs_before_stop=args.min_epochs_before_stop,
                    val_metric_smoothing=args.val_metric_smoothing,
                    class_weighting=args.class_weighting,
                )
            else:
                raise KeyError(model_key)
            row = _metric_row(ratio, model_key, metrics_path)
            rows.append(row)
            _print_metric_row(row)

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    test_summary_path = output_root / "test_summary.csv"
    test_fieldnames = ["ratio", "model", "category", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_inference_ms", "parameter_count", "metrics_path"]
    with test_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=test_fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in test_fieldnames} for row in rows])
    print(f"\n已写入基准模型对比实验汇总: {summary_path}", flush=True)
    print(f"已写入测试集效果汇总: {test_summary_path}", flush=True)


if __name__ == "__main__":
    main()

