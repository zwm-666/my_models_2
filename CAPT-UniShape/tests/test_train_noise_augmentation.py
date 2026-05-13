from __future__ import annotations

import unittest

import torch

from train import apply_relative_gaussian_noise


class TrainNoiseAugmentationTests(unittest.TestCase):
    def test_zero_noise_range_returns_input(self) -> None:
        sequence = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        noisy = apply_relative_gaussian_noise(sequence, std_min=0.0, std_max=0.0)
        self.assertTrue(torch.equal(noisy, sequence))

    def test_fixed_noise_scale_matches_requested_ratio(self) -> None:
        torch.manual_seed(1234)
        sequence = torch.linspace(-1.0, 1.0, steps=4000, dtype=torch.float32).reshape(1, 1, 4000)
        noisy = apply_relative_gaussian_noise(sequence, std_min=0.2, std_max=0.2)
        noise = noisy - sequence
        signal_std = float(sequence.std(unbiased=False))
        noise_std = float(noise.std(unbiased=False))
        self.assertAlmostEqual(noise_std / signal_std, 0.2, delta=0.03)


if __name__ == "__main__":
    unittest.main()
