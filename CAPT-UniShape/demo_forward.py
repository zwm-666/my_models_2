"""Forward demo for Official-CAPT-UniShape-RBF-KANFusion and No-RBF control."""

from __future__ import annotations

from importlib import import_module
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_models_module = import_module("models")
OfficialCAPTUniShapeKANFusionNoRBF = getattr(_models_module, "OfficialCAPTUniShapeKANFusionNoRBF")
OfficialCAPTUniShapeRBFKANFusion = getattr(_models_module, "OfficialCAPTUniShapeRBFKANFusion")


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def tensor_to_float(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().cpu())


def run_one(name: str, model: torch.nn.Module, x_op: torch.Tensor, x_eis: torch.Tensor, x_cond: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        logits, loss_dict = model(x_op, x_eis, x_cond, labels)
    elapsed = time.perf_counter() - start
    result = {
        "model": name,
        "logits_shape": list(logits.shape),
        "total_loss": tensor_to_float(loss_dict.get("total_loss")),
        "ce_loss": tensor_to_float(loss_dict.get("ce_loss")),
        "transport_loss": tensor_to_float(loss_dict.get("transport_loss")),
        "separation_loss": tensor_to_float(loss_dict.get("separation_loss")),
        "kan_regularization": tensor_to_float(loss_dict.get("kan_regularization")),
        "parameter_count": count_parameters(model),
        "forward_time_s": elapsed,
    }
    if loss_dict.get("delta") is not None:
        shifted_cond = x_cond + 0.5
        with torch.no_grad():
            shifted_logits, shifted_loss = model(x_op, x_eis, shifted_cond, labels)
        delta_change = torch.mean(torch.abs(shifted_loss["delta"] - loss_dict["delta"]))
        logits_change = torch.mean(torch.abs(shifted_logits - logits))
        result["condition_delta_change_mean_abs"] = tensor_to_float(delta_change)
        result["condition_logits_change_mean_abs"] = tensor_to_float(logits_change)
        if delta_change <= 0:
            raise RuntimeError("RBF dynamic prototypes did not change after condition perturbation")
    with torch.no_grad():
        op_changed_logits, _ = model(x_op * 1.01, x_eis, x_cond, labels)
        eis_changed_logits, _ = model(x_op, x_eis * 1.01, x_cond, labels)
    result["op_logits_change_mean_abs"] = tensor_to_float(torch.mean(torch.abs(op_changed_logits - logits)))
    result["eis_logits_change_mean_abs"] = tensor_to_float(torch.mean(torch.abs(eis_changed_logits - logits)))
    print(f"\n{name}")
    print(f"  logits shape: {tuple(logits.shape)}")
    print(f"  total loss: {result['total_loss']:.6f}")
    print(f"  CE loss: {result['ce_loss']:.6f}")
    print(f"  transport loss: {result['transport_loss']:.6f}")
    print(f"  parameters: {result['parameter_count']}")
    print(f"  forward time: {elapsed:.4f}s")
    return result


def main() -> None:
    torch.manual_seed(42)
    batch_size = 8
    c_op = 6
    op_len = 256
    c_eis = 4
    eis_len = 128
    d_cond = 10
    num_classes = 3
    config: dict[str, Any] = {
        "c_op": c_op,
        "op_seq_len": op_len,
        "c_eis": c_eis,
        "eis_seq_len": eis_len,
        "d_cond": d_cond,
        "num_classes": num_classes,
        "d_model": 128,
        "hidden_dim": 256,
        "fusion_hidden_dim": 256,
        "kan_bottleneck_dim": 32,
        "kan_num_basis": 8,
        "kan_lambda": 0.1,
        "channel_aggregation": "attention",
        "freeze_unishape_backbone": False,
        "use_residual_kan_fusion": True,
        "dropout": 0.1,
    }
    x_op = torch.randn(batch_size, c_op, op_len)
    x_eis = torch.randn(batch_size, c_eis, eis_len)
    x_cond = torch.randn(batch_size, d_cond)
    labels = torch.randint(0, num_classes, (batch_size,))

    model_a = OfficialCAPTUniShapeRBFKANFusion(**config)
    model_b = OfficialCAPTUniShapeKANFusionNoRBF(**config)
    results = [
        run_one("Official-CAPT-UniShape-RBF-KANFusion", model_a, x_op, x_eis, x_cond, labels),
        run_one("Official-CAPT-UniShape-KANFusion-NoRBF", model_b, x_op, x_eis, x_cond, labels),
    ]
    output_dir = Path("results/official_unishape_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "demo_forward_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
