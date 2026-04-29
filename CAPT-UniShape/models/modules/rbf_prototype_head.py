"""RBF condition-aware dynamic prototype classification head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFConditionMapper(nn.Module):
    """Map condition tokens to class-wise prototype offsets with RBF bases."""

    def __init__(
        self,
        cond_dim: int,
        num_classes: int,
        d_model: int,
        num_centers: int = 16,
        output_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.centers = nn.Parameter(torch.randn(int(num_centers), self.cond_dim) * 0.1)
        self.log_width = nn.Parameter(torch.zeros(int(num_centers)))
        self.linear = nn.Linear(int(num_centers), self.num_classes * self.d_model)
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))
        nn.init.trunc_normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z_cond: torch.Tensor) -> torch.Tensor:
        if z_cond.ndim != 2 or z_cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected [B, {self.cond_dim}] condition token, got {tuple(z_cond.shape)}")
        dist2 = (z_cond.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(dim=-1)
        width2 = torch.exp(self.log_width).clamp_min(1e-3).pow(2).unsqueeze(0)
        rbf = torch.exp(-dist2 / (2.0 * width2))
        delta = self.linear(rbf).view(-1, self.num_classes, self.d_model)
        return self.output_scale * delta


class RBFPrototypeHead(nn.Module):
    """Condition-aware RBF dynamic prototype head.

    Args:
        d_model: Dimension of fused representation ``h`` and prototypes.
        cond_dim: Dimension of condition token used by the RBF mapper.
        num_classes: Number of fault classes.
        temperature: Cosine-logit temperature.
    """

    def __init__(
        self,
        d_model: int,
        cond_dim: int,
        num_classes: int,
        temperature: float = 0.07,
        num_rbf_centers: int = 16,
        separation_margin: float = 0.2,
        use_condition_transport: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.separation_margin = float(separation_margin)
        self.use_condition_transport = bool(use_condition_transport)
        self.prototypes = nn.Parameter(torch.empty(self.num_classes, self.d_model))
        nn.init.xavier_uniform_(self.prototypes)
        self.mapper = RBFConditionMapper(
            cond_dim=int(cond_dim),
            num_classes=self.num_classes,
            d_model=self.d_model,
            num_centers=int(num_rbf_centers),
        )
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(0.01, 1.0)

    def forward(self, h: torch.Tensor, z_cond: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.ndim != 2 or h.shape[-1] != self.d_model:
            raise ValueError(f"Expected [B, {self.d_model}] representation, got {tuple(h.shape)}")
        if self.use_condition_transport:
            delta = self.mapper(z_cond)
        else:
            delta = h.new_zeros(h.shape[0], self.num_classes, self.d_model)
        dynamic_prototypes = self.prototypes.unsqueeze(0) + delta
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(dynamic_prototypes, dim=-1)
        logits = torch.einsum("bd,bkd->bk", h_norm, p_norm) / self.temperature
        aux = {
            "delta": delta,
            "dynamic_prototypes": dynamic_prototypes,
            "static_prototypes": self.prototypes,
            "loss_transport": self.transport_regularization(delta),
            "loss_separation": self.prototype_separation_loss(),
        }
        return logits, aux

    @staticmethod
    def transport_regularization(delta: torch.Tensor) -> torch.Tensor:
        return delta.pow(2).mean()

    def prototype_separation_loss(self) -> torch.Tensor:
        proto = F.normalize(self.prototypes, dim=-1)
        cosine = proto @ proto.T
        eye = torch.eye(self.num_classes, device=cosine.device, dtype=torch.bool)
        off_diag = cosine.masked_select(~eye)
        return F.relu(off_diag - self.separation_margin).mean()
