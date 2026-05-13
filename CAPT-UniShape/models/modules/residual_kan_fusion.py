"""Residual KAN-Fusion FFN for operation/EIS/condition features."""

from __future__ import annotations

import torch
import torch.nn as nn

from .simple_kan import SimpleKANLayer


class FeatureDropout(nn.Module):
    """Channel-style dropout for [B, D] features."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return x
        return nn.functional.dropout1d(x.unsqueeze(-1), p=self.p, training=self.training).squeeze(-1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(channels), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(channels)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.net(x)
        return x * weights, weights


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
        feature_dropout: float = 0.0,
        stochastic_depth_p: float = 0.0,
        cond_dim: int | None = None,
        se_hidden_dim: int = 32,
        use_residual_kan: bool = True,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or max(int(d_model) * 2, int(input_dim)))
        self.use_residual_kan = bool(use_residual_kan)
        self.use_film = bool(use_film)
        self.stochastic_depth_p = float(stochastic_depth_p)
        self.cond_width = int(cond_dim or d_model)
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.main_linear1 = nn.Linear(int(input_dim), hidden)
        self.main_act = nn.GELU()
        self.main_dropout = nn.Dropout(float(dropout))
        self.main_feature_dropout = FeatureDropout(float(feature_dropout))
        self.se_block = SEBlock(hidden, hidden_dim=int(se_hidden_dim))
        self.main_linear2 = nn.Linear(hidden, int(d_model))
        self.output_norm = nn.LayerNorm(int(d_model))
        if self.use_residual_kan:
            self.bottleneck: nn.Module | None = nn.Sequential(
                nn.Linear(int(input_dim), int(bottleneck_dim)),
                nn.LayerNorm(int(bottleneck_dim)),
                nn.Tanh(),
            )
            self.kan: SimpleKANLayer | None = SimpleKANLayer(
                input_dim=int(bottleneck_dim),
                output_dim=int(bottleneck_dim),
                num_basis=int(num_basis),
                dropout=float(dropout),
            )
            self.kan_to_model: nn.Linear | None = nn.Linear(int(bottleneck_dim), int(d_model))
        else:
            self.bottleneck = None
            self.kan = None
            self.kan_to_model = None
        if self.use_residual_kan and learnable_lambda:
            self.lambda_kan = nn.Parameter(torch.tensor(float(lambda_kan)))
        else:
            self.register_buffer("lambda_kan", torch.tensor(float(lambda_kan) if self.use_residual_kan else 0.0))
        if self.use_film:
            self.film_gamma: nn.Linear | None = nn.Linear(self.cond_width, int(d_model))
            self.film_beta: nn.Linear | None = nn.Linear(self.cond_width, int(d_model))
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.ones_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
        else:
            self.film_gamma = None
            self.film_beta = None

    def _apply_stochastic_depth(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.stochastic_depth_p <= 0.0:
            mask = residual.new_ones((residual.shape[0], 1))
            return residual, mask
        keep_prob = max(1e-6, 1.0 - self.stochastic_depth_p)
        mask = torch.bernoulli(residual.new_full((residual.shape[0], 1), keep_prob))
        return residual * mask / keep_prob, mask

    def forward(self, fusion_input: torch.Tensor, z_cond: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.input_norm(fusion_input)
        x = self.main_linear1(x)
        x = self.main_act(x)
        x = self.main_dropout(x)
        x = self.main_feature_dropout(x)
        x, se_weights = self.se_block(x)
        z_main = self.output_norm(self.main_linear2(x))
        if self.use_residual_kan:
            if self.bottleneck is None or self.kan is None or self.kan_to_model is None:
                raise RuntimeError("Residual KAN branch is enabled but KAN modules were not initialized")
            z_bottleneck = self.bottleneck(fusion_input)
            z_kan = self.kan_to_model(self.kan(z_bottleneck))
            z_kan, depth_mask = self._apply_stochastic_depth(z_kan)
            h = z_main + self.lambda_kan * z_kan
            kan_regularization = self.kan.regularization()
        else:
            z_kan = torch.zeros_like(z_main)
            depth_mask = torch.ones((z_main.shape[0], 1), device=z_main.device, dtype=z_main.dtype)
            h = z_main
            kan_regularization = fusion_input.new_zeros(())
        if z_cond is None:
            z_cond = fusion_input.new_zeros((fusion_input.shape[0], self.cond_width))
        if self.use_film:
            if self.film_gamma is None or self.film_beta is None:
                raise RuntimeError("FiLM is enabled but FiLM modules were not initialized")
            gamma = self.film_gamma(z_cond)
            beta = self.film_beta(z_cond)
        else:
            gamma = torch.ones_like(z_main)
            beta = torch.zeros_like(z_main)
        h = gamma * h + beta
        aux = {
            "z_main": z_main,
            "z_kan": z_kan,
            "film_gamma": gamma,
            "film_beta": beta,
            "se_weights": se_weights,
            "stochastic_depth_mask": depth_mask,
            "lambda_kan": self.lambda_kan.reshape(()),
            "kan_regularization": kan_regularization,
        }
        return h, aux
