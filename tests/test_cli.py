from __future__ import annotations

import json
import unittest

from xpu_benchmark.cli import _render_results, build_parser
from xpu_benchmark.harness import run_named_benchmarks
from xpu_benchmark.ops import BENCHMARK_SPECS


class CliTests(unittest.TestCase):
    def test_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run"])
        self.assertEqual(args.device, "xpu")
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.ops, sorted(BENCHMARK_SPECS))
        self.assertEqual(args.timer, "timeit")

    def test_parser_accepts_timeit_timer(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--timer", "timeit"])
        self.assertEqual(args.timer, "timeit")

    def test_parser_accepts_event_timer(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--timer", "event"])
        self.assertEqual(args.timer, "event")

    def test_parser_accepts_fixed_runs(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--runs", "7"])
        self.assertEqual(args.runs, 7)

    def test_parser_defaults_roofline_peaks(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run"])
        self.assertEqual(args.peak_tflops, 183.0)
        self.assertEqual(args.peak_bandwidth_gbps, 608.0)

    def test_json_results_are_serializable(self) -> None:
        _, results = run_named_benchmarks(
            op_names=["addmm"],
            device_name="cpu",
            dtype_name="float32",
            min_run_time=0.01,
        )
        rendered = json.dumps([result.to_dict() for result in results])
        self.assertIn("addmm", rendered)
        self.assertIn("input_shape", rendered)
        self.assertIn('"dtype_name": "float32"', rendered)
        self.assertIn("tflops", rendered)
        self.assertIn("bandwidth_gbps", rendered)
        self.assertIn("peak_tflops_percent", rendered)
        self.assertIn("peak_bandwidth_percent", rendered)
        self.assertIn("arithmetic_intensity", rendered)
        self.assertIn("roofline_tflops", rendered)
        self.assertIn("eff_vs_roofline_percent", rendered)
        self.assertIn("bound_hint", rendered)

    def test_timeit_results_are_serializable(self) -> None:
        measurements, results = run_named_benchmarks(
            op_names=["addmm"],
            device_name="cpu",
            dtype_name="float32",
            min_run_time=0.0,
            timer_backend="timeit",
        )
        self.assertEqual(measurements, [])
        rendered = json.dumps([result.to_dict() for result in results])
        self.assertIn('"timer_backend": "timeit"', rendered)
        self.assertIn('"input_shape": "mat1=(512, 512), mat2=(512, 512), bias=(512, 512)"', rendered)

    def test_fixed_runs_are_used_with_timeit_timer(self) -> None:
        _, results = run_named_benchmarks(
            op_names=["addmm"],
            device_name="cpu",
            dtype_name="float32",
            min_run_time=10.0,
            timer_backend="timeit",
            runs=3,
        )
        self.assertEqual(results[0].number_per_run, 3)

    def test_fixed_runs_are_used_with_torch_timer(self) -> None:
        _, results = run_named_benchmarks(
            op_names=["addmm"],
            device_name="cpu",
            dtype_name="float32",
            min_run_time=10.0,
            timer_backend="torch",
            runs=3,
        )
        self.assertEqual(results[0].number_per_run, 3)

    def test_text_results_include_input_shape_for_all_timers(self) -> None:
        result = {
            "op_name": "addmm",
            "input_shape": "mat1=(512, 512), mat2=(512, 512), bias=(512, 512)",
            "dtype_name": "float32",
            "timer_backend": "torch",
            "median_seconds": 1.0e-4,
            "mean_seconds": 1.1e-4,
            "number_per_run": 100,
            "tflops": 2.5,
            "peak_tflops_percent": 1.37,
            "bandwidth_gbps": 300.0,
            "peak_bandwidth_percent": 49.34,
            "arithmetic_intensity": 8.33,
            "roofline_tflops": 5.06,
            "eff_vs_roofline_percent": 49.41,
            "bound_hint": "memory-bound",
        }
        rendered = _render_results([], [result], "compare")
        self.assertIn("input shape", rendered)
        self.assertIn("dtype", rendered)
        self.assertIn("TFLOPS", rendered)
        self.assertIn("%peak_tflops", rendered)
        self.assertIn("GB/s", rendered)
        self.assertIn("%peak_bw", rendered)
        self.assertIn("AI(F/B)", rendered)
        self.assertIn("roofline_tflops", rendered)
        self.assertIn("eff_vs_roofline(%)", rendered)
        self.assertIn("bound_hint", rendered)
        self.assertIn("float32", rendered)
        self.assertIn("mat1=(512, 512)", rendered)
        self.assertIn("torch", rendered)
        self.assertIn("2.50", rendered)
        self.assertIn("300.00", rendered)
        self.assertIn("memory-bound", rendered)


if __name__ == "__main__":
    unittest.main()
