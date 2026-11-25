import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluation_utils import EvalConfig, iter_windows, save_results

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

    from evaluation_metrics import chunked_nll


class WindowTests(unittest.TestCase):
    def test_every_target_once(self):
        for context, stride in ((2, 1), (8, 1), (8, 3), (8, 7), (2048, 512)):
            for size in (2, 7, 8, 9, 11, 17, 2048, 2049, 2560, 2561, 8192):
                with self.subTest(context=context, stride=stride, size=size):
                    targets = []
                    for begin, end, first in iter_windows(size, context, stride):
                        self.assertLessEqual(end - begin, context)
                        self.assertGreaterEqual(first, begin + 1)
                        targets.extend(range(first, end))
                    self.assertEqual(targets, list(range(1, size)))

    def test_exact_boundary_and_tail(self):
        self.assertEqual(list(iter_windows(8, 8, 3)), [(0, 8, 1)])
        self.assertEqual(
            list(iter_windows(12, 8, 3)), [(0, 8, 1), (3, 11, 8), (6, 12, 11)]
        )

    def test_invalid_inputs(self):
        for args in ((1, 8, 3), (8, 1, 1), (8, 8, 0), (8, 8, 8)):
            with self.assertRaises(ValueError):
                list(iter_windows(*args))
        for config in (
            EvalConfig(repeats=0),
            EvalConfig(max_tokens=1),
            EvalConfig(pack_bits=16),
        ):
            with self.assertRaises(ValueError):
                config.validate()

    def test_results_and_ratios(self):
        row = {
            "length": 128,
            "median_ms": 4,
            "tokens_per_second": 32000,
            "samples_ms": [4],
        }
        report = {
            "status": "complete",
            "bf16": {"perplexity": {"ppl": 10}, "forward": [row]},
            "int4": {"perplexity": {"ppl": 12}, "forward": [dict(row, median_ms=8)]},
        }
        with tempfile.TemporaryDirectory() as directory:
            save_results(report, directory)
            self.assertEqual(
                json.loads(
                    (Path(directory) / "results.json").read_text(encoding="utf-8")
                ),
                report,
            )

            with (Path(directory) / "summary.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[1]["speedup_bf16_over_int4"]), 0.5)
            self.assertEqual(float(rows[1]["ppl_delta"]), 2)
            self.assertAlmostEqual(float(rows[1]["ppl_relative_change"]), 0.2)

    def test_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            save_results({"status": "failed", "stage": "bf16:forward"}, directory)
            self.assertTrue((Path(directory) / "results.json").is_file())


@unittest.skipUnless(TORCH_AVAILABLE, "CPU-проверки CE требуют PyTorch")
class LossTests(unittest.TestCase):
    def test_chunked_ce_matches_manual_shift(self):

        torch.manual_seed(0)
        logits = torch.randn(1, 12, 5)
        labels = torch.randint(0, 5, (1, 12))
        labels[:, :7] = -100
        expected = F.cross_entropy(
            logits[:, :-1].reshape(-1, 5),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        for chunk in (1, 3, 128):
            nll, count = chunked_nll(logits, labels, chunk)
            self.assertEqual(count, 5)
            self.assertAlmostEqual(nll, expected, places=5)

    def test_uniform_toy_model(self):

        class UniformModel(torch.nn.Module):
            def forward(self, ids):
                return torch.zeros((*ids.shape, 7))

        model = UniformModel()
        tokens = torch.arange(19).reshape(1, -1) % 7
        total, targets = 0.0, 0
        for begin, end, first in iter_windows(19, 8, 3):
            inputs = tokens[:, begin:end]
            labels = inputs.clone()
            labels[:, : first - begin] = -100
            nll, count = chunked_nll(model(inputs), labels, chunk_size=3)
            total += nll
            targets += count
        self.assertEqual(targets, 18)
        self.assertAlmostEqual(math.exp(total / targets), 7, places=5)

    def test_last_logit_is_excluded_with_partial_chunk(self):

        torch.manual_seed(0)
        for length in (2, 7, 12, 127, 128, 129, 2048):
            logits = torch.randn(2, length, 5)
            logits[:, -1, :] = float("nan")
            labels = torch.randint(0, 5, (2, length))
            expected = F.cross_entropy(
                logits[:, :-1].reshape(-1, 5),
                labels[:, 1:].reshape(-1),
                reduction="sum",
            ).item()
            for chunk in (1, 3, 128):
                with self.subTest(length=length, chunk=chunk):
                    nll, count = chunked_nll(logits, labels, chunk)
                    self.assertEqual(count, 2 * (length - 1))
                    self.assertTrue(math.isfinite(nll))
                    self.assertTrue(
                        math.isclose(nll, expected, rel_tol=1e-6, abs_tol=1e-5)
                    )


if __name__ == "__main__":
    unittest.main()
