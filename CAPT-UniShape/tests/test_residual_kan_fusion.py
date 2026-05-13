from __future__ import annotations

import unittest

import torch

from models.modules.residual_kan_fusion import ResidualKANFusion


class ResidualKANFusionTests(unittest.TestCase):
    def test_fusion_accepts_condition_for_gating_and_film(self) -> None:
        fusion = ResidualKANFusion(
            input_dim=96,
            d_model=96,
            hidden_dim=192,
            bottleneck_dim=32,
            num_basis=4,
            dropout=0.3,
            feature_dropout=0.3,
            stochastic_depth_p=0.2,
            cond_dim=96,
        )
        x = torch.randn(4, 96)
        z_cond = torch.randn(4, 96)

        h, aux = fusion(x, z_cond)

        self.assertEqual(tuple(h.shape), (4, 96))
        self.assertIn("film_gamma", aux)
        self.assertIn("film_beta", aux)
        self.assertIn("stochastic_depth_mask", aux)

    def test_disabled_film_uses_identity_modulation(self) -> None:
        fusion = ResidualKANFusion(
            input_dim=6,
            d_model=4,
            hidden_dim=8,
            bottleneck_dim=3,
            num_basis=4,
            cond_dim=4,
            use_film=False,
        )
        x = torch.randn(5, 6)
        z_cond = torch.randn(5, 4)

        _, aux = fusion(x, z_cond)

        self.assertTrue(torch.allclose(aux["film_gamma"], torch.ones(5, 4)))
        self.assertTrue(torch.allclose(aux["film_beta"], torch.zeros(5, 4)))


if __name__ == "__main__":
    unittest.main()
