"""Small CPU checks for the chunked oracle and report/figure plumbing."""
import csv
from pathlib import Path
import tempfile
import unittest

import torch

from ablation.sm90_flash_attention.plot import plot_results
from ablation.sm90_flash_attention.run import STAGES, oracle, table
from ablation.sm90_flash_attention.roofline import analytical_work, plot_roofline


class AttentionAblationTests(unittest.TestCase):
    def test_analytical_work(self):
        flops, io_bytes = analytical_work(512)
        self.assertEqual(flops, 4 * 4 * 8 * 128 * (512 * 513 // 2))
        self.assertEqual(io_bytes, 4 * 4 * 8 * 512 * 128 * 2 + 4 * 8 * 512 * 4)

    def test_uniform_causal_oracle(self):
        Q = torch.zeros(4, 7, 8, 128, dtype=torch.bfloat16)
        V = torch.arange(7, dtype=torch.float32)[None, :, None, None].expand_as(Q).bfloat16()
        O, lse = oracle(Q, Q, V)
        expected = (torch.arange(7, dtype=torch.float32) / 2)[None, :, None, None].expand_as(Q).bfloat16()
        torch.testing.assert_close(O, expected)
        torch.testing.assert_close(lse, torch.arange(1, 8, dtype=torch.float32).log()[None, None, :].expand(4, 8, 7))

    def test_table_and_plot(self):
        # Synthetic plotting-only fixture, never published as benchmark evidence.
        rows = [dict(sequence=T, stage=stage, technique=technique, custom_us=float(T / (index + 1)),
                     reference_us=float(T / 8), reference_pct=100 * (index + 1) / 8, speedup_a1=float(index + 1))
                for T in (512, 8192) for index, (stage, technique) in enumerate(STAGES.items())]
        report = table(rows)
        self.assertEqual(report.count("| bf16"), 16)
        self.assertIn("reference (FA4)", report)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "plot.png"
            plot_results(rows, destination)
            self.assertGreater(destination.stat().st_size, 1000)
            self.assertGreater(destination.with_suffix(".pdf").stat().st_size, 1000)
            plot_roofline(rows, destination.with_name("roofline.png"))
            self.assertGreater(destination.with_name("roofline.pdf").stat().st_size, 1000)
            with destination.with_name("roofline.csv").open() as file:
                points = list(csv.DictReader(file))
            self.assertEqual(len(points), 18)
            self.assertEqual(points[-1]["stage"], "FA4")


if __name__ == "__main__":
    unittest.main()
