"""Minimal KAN-style layer used as a research prototype fallback.

This is not a full pykan/efficient-kan reimplementation.  It keeps the key KAN
idea needed here: each scalar input is expanded on a learnable nonlinear grid
(Gaussian RBF bases), and the expanded bases are linearly mixed into the output.
For input dimension ``D``, output dimension ``O`` and ``M`` basis functions, the
dominant parameter count is ``D*M`` basis weights plus ``(D*M)*O`` mixer weights.
The dominant compute is ``O(B*D*M + B*(D*M)*O)`` for batch size ``B``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleKANLayer(nn.Module):
    """Batch-compatible KAN-style nonlinear layer for ``[B, D]`` inputs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_basis: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        dropout: float = 0.0,
        use_base_path: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_basis = int(num_basis)
        self.input_norm = nn.LayerNorm(self.input_dim)
        grid = torch.linspace(float(grid_min), float(grid_max), self.num_basis)
        self.centers = nn.Parameter(grid.repeat(self.input_dim, 1))
        self.log_width = nn.Parameter(torch.zeros(self.input_dim, self.num_basis))
        self.basis_weight = nn.Parameter(torch.empty(self.input_dim, self.num_basis))
        self.basis_mixer = nn.Linear(self.input_dim * self.num_basis, self.output_dim)
        self.base_path = nn.Linear(self.input_dim, self.output_dim) if use_base_path else None
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.basis_weight, std=0.05)
        nn.init.xavier_uniform_(self.basis_mixer.weight)
        nn.init.zeros_(self.basis_mixer.bias)
        if self.base_path is not None:
            nn.init.xavier_uniform_(self.base_path.weight)
            nn.init.zeros_(self.base_path.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [B, {self.input_dim}] input, got {tuple(x.shape)}")
        x_norm = self.input_norm(x)
        width = torch.exp(self.log_width).clamp_min(1e-3)
        basis = torch.exp(-((x_norm.unsqueeze(-1) - self.centers.unsqueeze(0)) / width.unsqueeze(0)).pow(2))
        weighted_basis = basis * self.basis_weight.unsqueeze(0)
        out = self.basis_mixer(self.dropout(weighted_basis.flatten(start_dim=1)))
        if self.base_path is not None:
            out = out + self.base_path(x)
        return out

    def regularization(self) -> torch.Tensor:
        """Small smoothness/scale penalty for the KAN basis parameters."""
        center_diff = self.centers[:, 1:] - self.centers[:, :-1]
        return self.basis_weight.pow(2).mean() + 0.01 * center_diff.pow(2).mean()
