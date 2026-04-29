"""Evaluate official CAPT-UniShape checkpoints and save paper-ready metrics."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

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
save_outputs = getattr(_train_module, "save_outputs")
sync_config_with_dataset = getattr(_train_module, "sync_config_with_dataset")


def run_evaluation(
    config_path: str,
    data_path: str,
    checkpoint_path: str,
    output_dir: str,
    strict: bool = True,
) -> dict[str, Any]:
    dataset = FuelCellNPZDataset(data_path)
    config = sync_config_with_dataset(load_config(config_path), dataset)
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
    save_outputs(Path(output_dir), metrics, count_parameters(model))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an official CAPT-UniShape checkpoint")
    parser.add_argument("--config", default="configs/rbf_kanfusion.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/official_unishape_eval")
    parser.add_argument("--allow-partial-load", action="store_true", help="Allow missing/unexpected checkpoint keys")
    args = parser.parse_args()
    run_evaluation(args.config, args.data, args.checkpoint, args.output_dir, strict=not args.allow_partial_load)


if __name__ == "__main__":
    main()
