"""Evaluate official CAPT-UniShape checkpoints and save paper-ready metrics."""

from __future__ import annotations

import argparse
import csv
import json
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_models_module = import_module("models")
_train_module = import_module("train")

build_model_from_config = getattr(_models_module, "build_model_from_config")
FuelCellNPZDataset = getattr(_train_module, "FuelCellNPZDataset")
count_parameters = getattr(_train_module, "count_parameters")
evaluate_loader = getattr(_train_module, "evaluate_loader")
load_config = getattr(_train_module, "load_config")
sync_config_with_dataset = getattr(_train_module, "sync_config_with_dataset")
_write_classification_report = getattr(_train_module, "_write_classification_report")
_write_confusion_matrix = getattr(_train_module, "_write_confusion_matrix")
_write_normalized_confusion_matrix = getattr(_train_module, "_write_normalized_confusion_matrix")
_write_predictions = getattr(_train_module, "_write_predictions")


def split_indices_from_npz(data: Mapping[str, Any], split_name: str) -> np.ndarray[Any, Any]:
    """Return sample indices for a named NPZ split.

    Split values follow the project convention: 0=train, 1=val, 2=test.
    """
    normalized = str(split_name).lower()
    if normalized in {"all", "full"}:
        if "split" in data:
            return np.arange(np.asarray(data["split"]).shape[0], dtype=np.int64)
        label_key = "labels" if "labels" in data else "y" if "y" in data else None
        if label_key is None:
            raise ValueError("Cannot infer sample count for split='all': NPZ data has no split/labels/y")
        return np.arange(np.asarray(data[label_key]).shape[0], dtype=np.int64)
    split_values = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}
    if normalized not in split_values:
        raise ValueError(f"Unsupported split: {split_name!r}. Use train, val, test or all.")
    if "split" not in data:
        raise ValueError(f"NPZ data must contain a split array when evaluating split={split_name!r}")
    split = np.asarray(data["split"], dtype=np.int64)
    indices = np.where(split == split_values[normalized])[0]
    if indices.size == 0:
        raise ValueError(f"NPZ data contains no samples for split={split_name!r}")
    return indices.astype(np.int64)


def normalize_split_name(split_name: str) -> str:
    normalized = str(split_name).lower()
    aliases = {"valid": "val", "validation": "val", "full": "all"}
    return aliases.get(normalized, normalized)


def evaluation_artifact_semantics(split_name: str) -> dict[str, str]:
    split = normalize_split_name(split_name)
    return {
        "top_level_metrics": split,
        "evaluated_split": split,
        "split_prefixed_files": f"{split}_prefixed_artifacts",
        "non_prefixed_confusion_matrix_csv": "not_written_by_evaluate_to_avoid_split_ambiguity",
    }


def save_evaluation_outputs(output_dir: Path, metrics: dict[str, Any], param_count: int, split_name: str) -> None:
    split = normalize_split_name(split_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_save = dict(metrics)
    metrics_to_save["parameter_count"] = int(param_count)
    metrics_to_save["artifact_semantics"] = evaluation_artifact_semantics(split)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_to_save, handle, indent=2, ensure_ascii=False)
    _write_confusion_matrix(output_dir / f"{split}_confusion_matrix.csv", metrics)
    _write_normalized_confusion_matrix(output_dir / f"{split}_confusion_matrix_normalized.csv", metrics)
    _write_predictions(output_dir / f"{split}_predictions.csv", metrics)
    _write_classification_report(output_dir / f"{split}_classification_report.csv", metrics)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            f"{split}_accuracy",
            f"{split}_macro_f1",
            f"{split}_weighted_f1",
            f"{split}_inference_ms",
            "parameter_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "split": split,
                f"{split}_accuracy": metrics_to_save["accuracy"],
                f"{split}_macro_f1": metrics_to_save["macro_f1"],
                f"{split}_weighted_f1": metrics_to_save["weighted_f1"],
                f"{split}_inference_ms": metrics_to_save["inference_time_per_sample_ms"],
                "parameter_count": param_count,
            }
        )


def run_evaluation(
    config_path: str,
    data_path: str,
    checkpoint_path: str,
    output_dir: str,
    strict: bool = True,
    split: str = "test",
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = np.load(data_path)
    indices = split_indices_from_npz(data, split)
    full_dataset = FuelCellNPZDataset(data_path)
    dataset = FuelCellNPZDataset(data_path, indices)
    raw_config = load_config(config_path)
    if config_overrides:
        raw_config.update(dict(config_overrides))
    config = sync_config_with_dataset(raw_config, full_dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    load_result = model.load_state_dict(state_dict, strict=strict)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Checkpoint load report: "
            f"missing_keys={list(load_result.missing_keys)}, "
            f"unexpected_keys={list(load_result.unexpected_keys)}"
        )
    batch_size = int(config.get("experiment", {}).get("batch_size", 8))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = evaluate_loader(model, loader, device, num_classes=int(config["num_classes"]))
    save_evaluation_outputs(Path(output_dir), metrics, count_parameters(model), split)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an official CAPT-UniShape checkpoint")
    parser.add_argument("--config", default="configs/rbf_kanfusion.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/official_unishape_eval")
    parser.add_argument("--split", default="test", choices=["train", "val", "valid", "validation", "test", "all"])
    parser.add_argument("--allow-partial-load", action="store_true", help="Allow missing/unexpected checkpoint keys")
    args = parser.parse_args()
    run_evaluation(args.config, args.data, args.checkpoint, args.output_dir, strict=not args.allow_partial_load, split=args.split)


if __name__ == "__main__":
    main()
