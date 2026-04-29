"""Default configuration for Residual KAN-Fusion research prototypes."""

from __future__ import annotations

import importlib
from typing import Any

_reskan = importlib.import_module("src.models.reskan")
build_model = _reskan.build_model


DEFAULT_RESKAN_CONFIG: dict[str, Any] = {
    "c_op": 6,
    "c_eis": 4,
    "d_cond": 12,
    "op_seq_len": 256,
    "eis_seq_len": 128,
    "d_model": 64,
    "fusion_hidden": 128,
    "kan_bottleneck": 32,
    "lambda_kan": 0.1,
    "learnable_lambda": False,
    "num_basis": 8,
    "num_classes": 3,
    "dropout": 0.1,
    "rbf_gamma": 1.0,
}


def get_config(**overrides: Any) -> dict[str, Any]:
    """Return a mutable config dict with optional overrides."""
    cfg = dict(DEFAULT_RESKAN_CONFIG)
    cfg.update(overrides)
    return cfg


def build_reskan_model(model_name: str, **overrides: Any):
    """Convenience factory used by the demo scripts."""
    return build_model(model_name, **get_config(**overrides))
