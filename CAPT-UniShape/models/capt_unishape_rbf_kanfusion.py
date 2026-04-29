"""Official-CAPT-UniShape-RBF-KANFusion model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import OfficialUniShapeBackboneWrapper
from .modules import ConditionEncoder, RBFPrototypeHead, ResidualKANFusion


class OfficialCAPTUniShapeRBFKANFusion(nn.Module):
    """Final enhanced model required by step12.

    Forward signature:
        ``forward(x_op, x_eis, x_cond, labels=None)`` where ``x_op`` is
        ``[B, C_op, T]``, ``x_eis`` is ``[B, C_eis, F]`` and ``x_cond`` is
        ``[B, D_cond]``.
    """

    def __init__(
        self,
        c_op: int = 6,
        c_eis: int = 4,
        d_cond: int = 10,
        op_seq_len: int = 256,
        eis_seq_len: int = 128,
        num_classes: int = 3,
        d_model: int = 128,
        hidden_dim: int = 256,
        fusion_hidden_dim: int | None = None,
        kan_bottleneck_dim: int = 32,
        kan_num_basis: int = 8,
        kan_lambda: float = 0.1,
        learnable_kan_lambda: bool = True,
        use_residual_kan_fusion: bool = True,
        dropout: float = 0.1,
        channel_aggregation: str = "attention",
        freeze_unishape_backbone: bool = False,
        scale_len: int = 3,
        temperature: float = 0.07,
        num_rbf_centers: int = 16,
        use_condition_transport: bool = True,
        alpha_transport: float = 1e-3,
        alpha_sep: float = 1e-3,
        alpha_kan: float = 1e-4,
        label_smoothing: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        del c_op, c_eis
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.alpha_transport = float(alpha_transport)
        self.alpha_sep = float(alpha_sep)
        self.alpha_kan = float(alpha_kan)
        self.label_smoothing = float(label_smoothing)

        self.op_backbone = OfficialUniShapeBackboneWrapper(
            series_size=int(op_seq_len),
            d_model=self.d_model,
            num_classes=self.num_classes,
            channel_aggregation=channel_aggregation,
            freeze_unishape_backbone=freeze_unishape_backbone,
            scale_len=int(scale_len),
            dropout=float(dropout),
        )
        self.eis_backbone = OfficialUniShapeBackboneWrapper(
            series_size=int(eis_seq_len),
            d_model=self.d_model,
            num_classes=self.num_classes,
            channel_aggregation=channel_aggregation,
            freeze_unishape_backbone=freeze_unishape_backbone,
            scale_len=int(scale_len),
            dropout=float(dropout),
        )
        self.condition_encoder = ConditionEncoder(
            d_cond=int(d_cond),
            d_model=self.d_model,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self.fusion = ResidualKANFusion(
            input_dim=self.d_model * 3,
            d_model=self.d_model,
            hidden_dim=fusion_hidden_dim,
            bottleneck_dim=int(kan_bottleneck_dim),
            num_basis=int(kan_num_basis),
            lambda_kan=float(kan_lambda),
            learnable_lambda=bool(learnable_kan_lambda),
            dropout=float(dropout),
            use_residual_kan=bool(use_residual_kan_fusion),
        )
        self.rbf_head = RBFPrototypeHead(
            d_model=self.d_model,
            cond_dim=self.d_model,
            num_classes=self.num_classes,
            temperature=float(temperature),
            num_rbf_centers=int(num_rbf_centers),
            use_condition_transport=bool(use_condition_transport),
        )

    def forward(
        self,
        x_op: torch.Tensor,
        x_eis: torch.Tensor,
        x_cond: torch.Tensor,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        z_op = self.op_backbone.extract_feature(x_op)
        z_eis = self.eis_backbone.extract_feature(x_eis)
        z_cond = self.condition_encoder(x_cond)
        fusion_input = torch.cat([z_op, z_eis, z_cond], dim=-1)
        h, fusion_aux = self.fusion(fusion_input)
        # Fair No-RBF ablation: both variants classify from the same fused
        # representation ``h``.  The RBF model receives condition information
        # only through dynamic prototype offsets, not an extra direct ``h+cond``
        # shortcut.
        logits, head_aux = self.rbf_head(h, z_cond)

        loss_dict: dict[str, torch.Tensor | None] = {
            "total_loss": None,
            "ce_loss": None,
            "transport_loss": head_aux["loss_transport"],
            "separation_loss": head_aux["loss_separation"],
            "kan_regularization": fusion_aux["kan_regularization"],
            "z_op": z_op,
            "z_eis": z_eis,
            "z_cond": z_cond,
            "h": h,
            "delta": head_aux["delta"],
            "dynamic_prototypes": head_aux["dynamic_prototypes"],
            "static_prototypes": head_aux["static_prototypes"],
            "lambda_kan": fusion_aux["lambda_kan"],
        }
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=self.label_smoothing)
            total_loss = (
                ce_loss
                + self.alpha_transport * head_aux["loss_transport"]
                + self.alpha_sep * head_aux["loss_separation"]
                + self.alpha_kan * fusion_aux["kan_regularization"]
            )
            loss_dict["ce_loss"] = ce_loss
            loss_dict["total_loss"] = total_loss
        return logits, loss_dict

    def load_official_unishape_weights(self, op_checkpoint: str | None = None, eis_checkpoint: str | None = None) -> dict[str, tuple[list[str], list[str]]]:
        """Load official UniShape checkpoints into operation/EIS wrappers."""
        report: dict[str, tuple[list[str], list[str]]] = {}
        if op_checkpoint:
            report["operation"] = self.op_backbone.load_pretrained(op_checkpoint, strict=False)
        if eis_checkpoint:
            report["eis"] = self.eis_backbone.load_pretrained(eis_checkpoint, strict=False)
        return report
