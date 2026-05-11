from __future__ import annotations

import unittest

import torch

from xpu_benchmark.ops import BENCHMARK_SPECS


class BenchmarkSpecTests(unittest.TestCase):
    def test_all_requested_benchmarks_exist(self) -> None:
        expected = {
            "addmm",
            "bmm",
            "group_gemm",
            "layernorm",
            "sum",
            "concat",
            "copy",
            "fused_attention_score",
        }
        self.assertEqual(set(BENCHMARK_SPECS), expected)

    def test_group_gemm_builds_and_runs_on_cpu(self) -> None:
        runner = BENCHMARK_SPECS["group_gemm"].build(torch.device("cpu"), torch.float32)
        result = runner()
        self.assertEqual(tuple(result.shape), (256, 256))


if __name__ == "__main__":
    unittest.main()
