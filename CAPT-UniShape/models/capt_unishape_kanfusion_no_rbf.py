"""Official-CAPT-UniShape-KANFusion-NoRBF control model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import OfficialUniShapeBackboneWrapper
from .modules import ConditionEncoder, ResidualKANFusion


class OfficialCAPTUniShapeKANFusionNoRBF(nn.Module):
    """Fair No-RBF control sharing the same backbone/condition/fusion trunk."""

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
        classifier_hidden_dim: int | None = None,
        kan_bottleneck_dim: int = 32,
        kan_num_basis: int = 8,
        kan_lambda: float = 0.1,
        learnable_kan_lambda: bool = True,
        use_residual_kan_fusion: bool = True,
        use_condition_gating: bool = True,
        use_film_modulation: bool = True,
        dropout: float = 0.1,
        feature_dropout: float = 0.0,
        stochastic_depth_p: float = 0.0,
        mixup_alpha: float = 0.0,
        se_hidden_dim: int = 32,
        channel_aggregation: str = "attention",
        freeze_unishape_backbone: bool = False,
        scale_len: int = 3,
        alpha_kan: float = 1e-4,
        label_smoothing: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        del c_op, c_eis
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.alpha_kan = float(alpha_kan)
        self.label_smoothing = float(label_smoothing)
        self.mixup_alpha = float(mixup_alpha)
        self.use_condition_gating = bool(use_condition_gating)
        classifier_hidden = int(classifier_hidden_dim or hidden_dim)

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
        self.op_gate = nn.Linear(self.d_model, self.d_model)
        self.eis_gate = nn.Linear(self.d_model, self.d_model)
        self.fusion = ResidualKANFusion(
            input_dim=self.d_model,
            d_model=self.d_model,
            hidden_dim=fusion_hidden_dim,
            bottleneck_dim=int(kan_bottleneck_dim),
            num_basis=int(kan_num_basis),
            lambda_kan=float(kan_lambda),
            learnable_lambda=bool(learnable_kan_lambda),
            dropout=float(dropout),
            feature_dropout=float(feature_dropout),
            stochastic_depth_p=float(stochastic_depth_p),
            cond_dim=self.d_model,
            se_hidden_dim=int(se_hidden_dim),
            use_residual_kan=bool(use_residual_kan_fusion),
            use_film=bool(use_film_modulation),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, classifier_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(classifier_hidden, self.num_classes),
        )

    def _mix_encoded_features(
        self,
        z_op: torch.Tensor,
        z_eis: torch.Tensor,
        z_cond: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, float | None]:
        if labels is None or not self.training or self.mixup_alpha <= 0.0 or z_op.shape[0] < 2:
            return z_op, z_eis, z_cond, None, None
        beta_dist = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        lam = float(beta_dist.sample().item())
        perm = torch.randperm(z_op.shape[0], device=z_op.device)
        z_op = lam * z_op + (1.0 - lam) * z_op[perm]
        z_eis = lam * z_eis + (1.0 - lam) * z_eis[perm]
        z_cond = lam * z_cond + (1.0 - lam) * z_cond[perm]
        return z_op, z_eis, z_cond, labels[perm], lam

    def _fuse_modal_features(
        self,
        z_op: torch.Tensor,
        z_eis: torch.Tensor,
        z_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_condition_gating:
            g_op = torch.ones_like(z_op)
            g_eis = torch.ones_like(z_eis)
            return z_op + z_eis + z_cond, g_op, g_eis
        g_op = torch.sigmoid(self.op_gate(z_cond))
        g_eis = torch.sigmoid(self.eis_gate(z_cond))
        return g_op * z_op + g_eis * z_eis + z_cond, g_op, g_eis

    def forward(
        self,
        x_op: torch.Tensor,
        x_eis: torch.Tensor,
        x_cond: torch.Tensor,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        logit_adjustment: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        z_op = self.op_backbone.extract_feature(x_op)
        z_eis = self.eis_backbone.extract_feature(x_eis)
        z_cond = self.condition_encoder(x_cond)
        z_op, z_eis, z_cond, mixup_labels_b, mixup_lambda = self._mix_encoded_features(z_op, z_eis, z_cond, labels)
        z_fused, g_op, g_eis = self._fuse_modal_features(z_op, z_eis, z_cond)
        h, fusion_aux = self.fusion(z_fused, z_cond)
        logits = self.classifier(h)
        loss_dict: dict[str, torch.Tensor | None] = {
            "total_loss": None,
            "ce_loss": None,
            "kan_regularization": fusion_aux["kan_regularization"],
            "transport_loss": torch.zeros((), device=h.device),
            "separation_loss": torch.zeros((), device=h.device),
            "z_op": z_op,
            "z_eis": z_eis,
            "z_cond": z_cond,
            "g_op": g_op,
            "g_eis": g_eis,
            "z_fused": z_fused,
            "h": h,
            "lambda_kan": fusion_aux["lambda_kan"],
            "mixup_lambda": None if mixup_lambda is None else torch.tensor(mixup_lambda, device=h.device),
        }
        if labels is not None:
            loss_logits = logits if logit_adjustment is None else logits + logit_adjustment.to(logits.device)
            if mixup_labels_b is None or mixup_lambda is None:
                ce_loss = F.cross_entropy(loss_logits, labels, weight=class_weights, label_smoothing=self.label_smoothing)
            else:
                ce_loss_a = F.cross_entropy(loss_logits, labels, weight=class_weights, label_smoothing=self.label_smoothing)
                ce_loss_b = F.cross_entropy(
                    loss_logits,
                    mixup_labels_b.to(labels.device),
                    weight=class_weights,
                    label_smoothing=self.label_smoothing,
                )
                ce_loss = float(mixup_lambda) * ce_loss_a + (1.0 - float(mixup_lambda)) * ce_loss_b
            loss_dict["ce_loss"] = ce_loss
            loss_dict["total_loss"] = ce_loss + self.alpha_kan * fusion_aux["kan_regularization"]
        return logits, loss_dict

    def load_official_unishape_weights(self, op_checkpoint: str | None = None, eis_checkpoint: str | None = None) -> dict[str, tuple[list[str], list[str]]]:
        report: dict[str, tuple[list[str], list[str]]] = {}
        if op_checkpoint:
            report["operation"] = self.op_backbone.load_pretrained(op_checkpoint, strict=False)
        if eis_checkpoint:
            report["eis"] = self.eis_backbone.load_pretrained(eis_checkpoint, strict=False)
        return report
