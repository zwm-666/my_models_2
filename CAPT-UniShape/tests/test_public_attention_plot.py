from __future__ import annotations

import unittest

import torch


class PublicAttentionPlotTests(unittest.TestCase):
    def test_channel_summary_averages_attention_by_class_and_channel(self) -> None:
        from scripts.plot_public_attention_maps import summarize_channel_attention

        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        weights = torch.tensor(
            [
                [0.8, 0.2],
                [0.6, 0.4],
                [0.1, 0.9],
                [0.3, 0.7],
            ],
            dtype=torch.float32,
        )

        summary = summarize_channel_attention(weights, labels)

        self.assertEqual(summary["class_ids"], [0, 1])
        self.assertEqual(summary["matrix"].shape, (2, 2))
        self.assertAlmostEqual(float(summary["matrix"][0, 0]), 0.7, places=5)
        self.assertAlmostEqual(float(summary["matrix"][1, 1]), 0.8, places=5)

    def test_attention_plot_cli_accepts_self_dataset_labels_without_changing_public_defaults(self) -> None:
        from scripts.plot_public_attention_maps import parse_args

        default_args = parse_args(["--data", "sample.npz", "--checkpoint", "sample.ckpt"])

        self.assertEqual(default_args.output_prefix, "public_ac_voltage")
        self.assertEqual(default_args.title_prefix, "Public AC Voltage")

        self_args = parse_args(
            [
                "--data",
                "self.npz",
                "--checkpoint",
                "self.ckpt",
                "--output-prefix",
                "self",
                "--title-prefix",
                "Self-Measured PEMFC",
                "--op-channel-labels",
                "stack_voltage,stack_current,stack_power",
                "--eis-channel-labels",
                "impedance,gradient,cumulative,freq_axis",
            ]
        )

        self.assertEqual(self_args.output_prefix, "self")
        self.assertEqual(self_args.title_prefix, "Self-Measured PEMFC")
        self.assertEqual(self_args.op_channel_labels, "stack_voltage,stack_current,stack_power")
        self.assertEqual(self_args.eis_channel_labels, "impedance,gradient,cumulative,freq_axis")


if __name__ == "__main__":
    unittest.main()
