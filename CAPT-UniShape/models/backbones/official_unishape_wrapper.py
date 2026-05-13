"""Official UniShape backbone wrapper for multi-source PEMFC inputs.

Input tensors are shaped ``[B, C, L]``.  Official UniShape is univariate by
design, so this wrapper keeps the upstream TokenGeneratorUnit, InceptionModule,
TransformerEnc, attention_head and fc_token_shape logic intact and applies the
same shared UniShape encoder independently to each channel when ``C > 1``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from external.unishape.models.unishapemodel_finetune import UniShapeModel


class OfficialUniShapeBackboneWrapper(nn.Module):
    """Feature-only wrapper around the official UniShape fine-tune model.

    Args:
        series_size: Input sequence length ``L`` expected by official UniShape.
        d_model: Output feature dimension returned by ``extract_feature``.
        num_classes: Class count used only to initialise the official fine-tune
            classification head; downstream CAPT heads are defined separately.
        channel_aggregation: Aggregation used for multichannel inputs.  The
            default ``attention`` computes softmax channel weights from shared
            UniShape features.  ``learnable_weighted`` uses channel-index logits
            and ``mean`` averages channels.
        freeze_unishape_backbone: If true, all official UniShape parameters are
            frozen while the channel aggregation/projection remains trainable.
        scale_len: Official scale index in ``[1, 5]``; the upstream code maps it
            to window sizes ``[64, 32, 16, 8, 4]``.
    """

    def __init__(
        self,
        series_size: int,
        d_model: int = 128,
        num_classes: int = 3,
        channel_aggregation: str = "attention",
        freeze_unishape_backbone: bool = False,
        scale_len: int = 3,
        shape_ratio: float = 0.25,
        official_hidden_dim: int = 128,
        max_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if series_size < 64:
            raise ValueError("Official UniShape wrapper expects series_size >= 64 for the default scales")
        if channel_aggregation not in {"mean", "attention", "learnable_weighted"}:
            raise ValueError(f"Unsupported channel_aggregation: {channel_aggregation}")
        if not 1 <= scale_len <= 5:
            raise ValueError("scale_len must be in [1, 5]")

        self.series_size = int(series_size)
        self.d_model = int(d_model)
        self.official_hidden_dim = int(official_hidden_dim)
        self.channel_aggregation = channel_aggregation
        self.max_channels = int(max_channels)

        official_config = SimpleNamespace(window_size=16, stride=16)
        self.unishape_backbone = UniShapeModel(
            official_config,
            series_size=self.series_size,
            in_channels=self.official_hidden_dim,
            window_emb_dim=self.official_hidden_dim,
            out_channels=int(num_classes),
            shape_ratio=float(shape_ratio),
            scale_len=int(scale_len),
            dropout=float(dropout),
        )

        self.output_projection = nn.Identity()
        if self.official_hidden_dim != self.d_model:
            self.output_projection = nn.Sequential(
                nn.LayerNorm(self.official_hidden_dim),
                nn.Linear(self.official_hidden_dim, self.d_model),
            )

        self.channel_attention = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, max(self.d_model // 2, 16)),
            nn.Tanh(),
            nn.Linear(max(self.d_model // 2, 16), 1),
        )
        self.channel_logits = nn.Parameter(torch.zeros(self.max_channels))

        if freeze_unishape_backbone:
            for parameter in self.unishape_backbone.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for ``extract_feature`` so the wrapper behaves as a backbone."""
        return self.extract_feature(x)

    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        feature, _ = self.extract_feature_with_attention(x)
        return feature

    def extract_feature_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a feature vector shaped ``[B, d_model]`` from ``x``.

        ``x`` must be ``[B, C, L]``.  ``C == 1`` is sent directly to official
        UniShape; ``C > 1`` is encoded channel-independently with shared weights
        and then aggregated.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, L] input, got {tuple(x.shape)}")
        _, channels, length = x.shape
        if length != self.series_size:
            raise ValueError(f"Expected sequence length {self.series_size}, got {length}")
        if channels < 1:
            raise ValueError("Input must contain at least one channel")

        if channels == 1:
            feature, temporal_attention = self._extract_univariate_feature_with_attention(x)
            aux = {
                "channel_weights": torch.ones((feature.shape[0], 1), device=feature.device, dtype=feature.dtype),
                "channel_features": feature.unsqueeze(1),
                "temporal_attention": temporal_attention.unsqueeze(1),
            }
            return feature, aux

        # UniShape is univariate, but running one Python/model call per channel
        # is very slow on CPU.  Folding channels into the batch dimension keeps
        # the exact same shared univariate encoder semantics while replacing C
        # serial backbone calls with one larger batched call.
        batch = int(x.shape[0])
        batched_channels = x.contiguous().view(batch * channels, 1, length)
        channel_features, temporal_attention = self._extract_univariate_feature_with_attention(batched_channels)
        stacked = channel_features.view(batch, channels, -1)  # [B, C, d_model]
        temporal_attention = temporal_attention.view(batch, channels, -1)
        aggregated, channel_weights = self._aggregate_channels_with_weights(stacked)
        aux = {
            "channel_weights": channel_weights,
            "channel_features": stacked,
            "temporal_attention": temporal_attention,
        }
        return aggregated, aux

    def _extract_univariate_feature(self, x: torch.Tensor) -> torch.Tensor:
        feature, _ = self._extract_univariate_feature_with_attention(x)
        return feature

    def _extract_univariate_feature_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.unishape_backbone
        window_size_list = [64, 32, 16, 8, 4]
        scale_index = model.scale_len - 1
        _ = window_size_list[scale_index]

        if scale_index == 4:
            x_embed = model.unit_scale_list[scale_index](x)
        else:
            x_embed = model.unit_scale_list_finetune[scale_index](x)

        cls_incep_tokens = model.inceptime_token(x_embed.permute(0, 2, 1)).permute(0, 2, 1)
        cls_incep_tokens = model.drop_token(model.layer_norm_inc(cls_incep_tokens))
        cls_incep_tokens = model.act_gelu_inc(cls_incep_tokens)
        attention_scores = model.attention_head(cls_incep_tokens)
        cls_tokens = torch.mean(cls_incep_tokens * attention_scores, dim=1).unsqueeze(1)

        x_embed_seq = x_embed.squeeze(1)
        if x_embed_seq.ndim == 2:
            x_embed_seq = x_embed_seq.unsqueeze(1)
        trans_enc_class_token, _shape_tokens = model.transformer_enc(x_embed_seq, cls_token_in=cls_tokens)
        feature = model.fc_token_shape(trans_enc_class_token)
        return self.output_projection(feature), attention_scores.squeeze(-1)

    def _aggregate_channels(self, features: torch.Tensor) -> torch.Tensor:
        aggregated, _ = self._aggregate_channels_with_weights(features)
        return aggregated

    def _aggregate_channels_with_weights(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channels = features.shape[1]
        if self.channel_aggregation == "mean":
            weights = torch.full(
                (features.shape[0], channels),
                1.0 / max(channels, 1),
                device=features.device,
                dtype=features.dtype,
            )
            return features.mean(dim=1), weights
        if self.channel_aggregation == "learnable_weighted":
            if channels > self.max_channels:
                raise ValueError(f"Got {channels} channels, but max_channels={self.max_channels}")
            weights = torch.softmax(self.channel_logits[:channels], dim=0)
            expanded = weights.unsqueeze(0).expand(features.shape[0], -1)
            return torch.einsum("bcd,c->bd", features, weights), expanded

        scores = self.channel_attention(features).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bcd,bc->bd", features, weights), weights

    def load_pretrained(self, checkpoint_path: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
        """Load official UniShape weights into the wrapped backbone.

        The loader accepts checkpoints containing either a raw state dict or a
        ``model_state_dict``/``state_dict`` key.  Prefixes from common training
        wrappers are stripped before loading.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            raise ValueError("Checkpoint must be a state-dict-like object")
        cleaned = {}
        for key, value in state_dict.items():
            clean_key = key
            for prefix in ("module.", "model.", "unishape_backbone."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            cleaned[clean_key] = value
        result = self.unishape_backbone.load_state_dict(cleaned, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)
