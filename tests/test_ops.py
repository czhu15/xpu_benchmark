from __future__ import annotations

import unittest

import torch

from xpu_benchmark.ops import BENCHMARK_CASES, BENCHMARK_SPECS


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
        self.assertEqual(set(BENCHMARK_CASES), expected)

    def test_each_benchmark_has_configured_cases(self) -> None:
        for op_name, spec in BENCHMARK_SPECS.items():
            self.assertGreaterEqual(len(spec.cases), 1, op_name)
            self.assertIsInstance(spec.cases[0], dict)

    def test_group_gemm_builds_and_runs_on_cpu(self) -> None:
        spec = BENCHMARK_SPECS["group_gemm"]
        runner = spec.build(torch.device("cpu"), torch.float32, spec.cases[0])
        result = runner()
        self.assertEqual(tuple(result.shape), (256, 256))


if __name__ == "__main__":
    unittest.main()
