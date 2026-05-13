from __future__ import annotations

import unittest

import torch


class PublicInterpretabilityTests(unittest.TestCase):
    def test_backbone_extract_feature_with_attention_returns_normalized_channel_weights(self) -> None:
        from models.backbones.official_unishape_wrapper import OfficialUniShapeBackboneWrapper

        torch.manual_seed(7)
        model = OfficialUniShapeBackboneWrapper(
            series_size=64,
            d_model=32,
            num_classes=3,
            channel_aggregation="attention",
            scale_len=3,
            dropout=0.0,
        )
        x = torch.randn(2, 3, 64)

        feature, aux = model.extract_feature_with_attention(x)

        self.assertEqual(tuple(feature.shape), (2, 32))
        self.assertEqual(tuple(aux["channel_weights"].shape), (2, 3))
        self.assertTrue(torch.allclose(aux["channel_weights"].sum(dim=1), torch.ones(2), atol=1e-5))
        self.assertEqual(aux["temporal_attention"].ndim, 3)
        self.assertEqual(aux["temporal_attention"].shape[0], 2)
        self.assertEqual(aux["temporal_attention"].shape[1], 3)

    def test_public_model_forward_exposes_attention_auxiliaries(self) -> None:
        from models.capt_unishape_rbf_kanfusion import OfficialCAPTUniShapeRBFKANFusion

        torch.manual_seed(11)
        model = OfficialCAPTUniShapeRBFKANFusion(
            c_op=3,
            c_eis=4,
            d_cond=12,
            op_seq_len=64,
            eis_seq_len=128,
            num_classes=3,
            d_model=32,
            hidden_dim=64,
            fusion_hidden_dim=64,
            kan_bottleneck_dim=8,
            kan_num_basis=4,
            dropout=0.0,
        )
        x_op = torch.randn(2, 3, 64)
        x_eis = torch.randn(2, 4, 128)
        x_cond = torch.randn(2, 12)

        logits, aux = model(x_op, x_eis, x_cond)

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertIn("op_channel_weights", aux)
        self.assertIn("eis_channel_weights", aux)
        self.assertIn("op_temporal_attention", aux)
        self.assertIn("eis_temporal_attention", aux)
        self.assertEqual(tuple(aux["op_channel_weights"].shape), (2, 3))
        self.assertEqual(tuple(aux["eis_channel_weights"].shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
