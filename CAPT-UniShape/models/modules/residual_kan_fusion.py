"""Residual KAN-Fusion FFN for operation/EIS/condition features."""

from __future__ import annotations

import torch
import torch.nn as nn

from .simple_kan import SimpleKANLayer


class ResidualKANFusion(nn.Module):
    """Fuse ``[z_op, z_eis, z_cond]`` with a stable MLP plus KAN branch.

    The KAN branch first compresses the concatenated feature to a bottleneck so
    KAN never processes the full high-dimensional fusion input directly:

    ``h = MLP(x) + lambda_kan * Linear(KAN(Bottleneck(x)))``.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        hidden_dim: int | None = None,
        bottleneck_dim: int = 32,
        num_basis: int = 8,
        lambda_kan: float = 0.1,
        learnable_lambda: bool = True,
        dropout: float = 0.1,
        use_residual_kan: bool = True,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or max(int(d_model) * 2, int(input_dim)))
        self.use_residual_kan = bool(use_residual_kan)
        self.main = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(d_model)),
            nn.LayerNorm(int(d_model)),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(int(input_dim), int(bottleneck_dim)),
            nn.LayerNorm(int(bottleneck_dim)),
            nn.Tanh(),
        )
        self.kan = SimpleKANLayer(
            input_dim=int(bottleneck_dim),
            output_dim=int(bottleneck_dim),
            num_basis=int(num_basis),
            dropout=float(dropout),
        )
        self.kan_to_model = nn.Linear(int(bottleneck_dim), int(d_model))
        if learnable_lambda:
            self.lambda_kan = nn.Parameter(torch.tensor(float(lambda_kan)))
        else:
            self.register_buffer("lambda_kan", torch.tensor(float(lambda_kan)))

    def forward(self, fusion_input: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z_main = self.main(fusion_input)
        z_bottleneck = self.bottleneck(fusion_input)
        z_kan = self.kan_to_model(self.kan(z_bottleneck))
        if self.use_residual_kan:
            h = z_main + self.lambda_kan * z_kan
        else:
            h = z_main
        aux = {
            "z_main": z_main,
            "z_kan": z_kan,
            "lambda_kan": self.lambda_kan.reshape(()),
            "kan_regularization": self.kan.regularization(),
        }
        return h, aux
