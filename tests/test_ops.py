from __future__ import annotations

import unittest

import torch

from xpu_benchmark.ops import BENCHMARK_CASES, BENCHMARK_SPECS


BASE_EXPECTED_SPECS = {
    "addmm",
    "bmm",
    "group_gemm",
    "layernorm",
    "sum",
    "concat",
    "copy",
    "fused_attention_score",
}

FORWARD_ONLY_EXPECTED_SPECS = {"triton_flash_attention", "triton_swiglu"}
BACKWARD_EXPECTED_SPECS = {f"{name}_backward" for name in BASE_EXPECTED_SPECS}


class BenchmarkSpecTests(unittest.TestCase):
    def test_all_requested_benchmarks_exist(self) -> None:
        self.assertEqual(set(BENCHMARK_SPECS), BASE_EXPECTED_SPECS | BACKWARD_EXPECTED_SPECS | FORWARD_ONLY_EXPECTED_SPECS)
        self.assertEqual(set(BENCHMARK_CASES), BASE_EXPECTED_SPECS | FORWARD_ONLY_EXPECTED_SPECS)

    def test_each_benchmark_has_configured_cases(self) -> None:
        for op_name, spec in BENCHMARK_SPECS.items():
            self.assertGreaterEqual(len(spec.cases), 1, op_name)
            self.assertIsInstance(spec.cases[0], dict)

    def test_each_backward_benchmark_builds_and_runs_on_cpu(self) -> None:
        device = torch.device("cpu")
        for op_name in sorted(BACKWARD_EXPECTED_SPECS):
            spec = BENCHMARK_SPECS[op_name]
            base_op_name = op_name.removesuffix("_backward")
            self.assertEqual(spec.cases, BENCHMARK_CASES[base_op_name], op_name)
            runner = spec.build(device, torch.float32, spec.cases[0])
            result = runner()
            self.assertIsInstance(result, torch.Tensor, op_name)
            self.assertEqual(result.device.type, "cpu", op_name)

    def test_group_gemm_builds_and_runs_on_cpu(self) -> None:
        spec = BENCHMARK_SPECS["group_gemm"]
        runner = spec.build(torch.device("cpu"), torch.float32, spec.cases[0])
        result = runner()
        self.assertEqual(tuple(result.shape), (256, 256))


if __name__ == "__main__":
    unittest.main()
