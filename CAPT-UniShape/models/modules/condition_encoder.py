"""Operating-condition encoder for official CAPT-UniShape models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Encode stack operating conditions into ``[B, d_model]`` tokens."""

    def __init__(
        self,
        d_cond: int,
        d_model: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or max(d_model, d_cond * 2))
        self.net = nn.Sequential(
            nn.Linear(int(d_cond), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(d_model)),
        )

    def forward(self, x_cond: torch.Tensor) -> torch.Tensor:
        if x_cond.ndim != 2:
            raise ValueError(f"Expected [B, D_cond] condition input, got {tuple(x_cond.shape)}")
        return self.net(x_cond)
