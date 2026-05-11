from __future__ import annotations

import json
import unittest

from xpu_benchmark.cli import build_parser
from xpu_benchmark.harness import run_named_benchmarks
from xpu_benchmark.ops import BENCHMARK_SPECS


class CliTests(unittest.TestCase):
    def test_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run"])
        self.assertEqual(args.device, "xpu")
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(args.ops, sorted(BENCHMARK_SPECS))

    def test_json_results_are_serializable(self) -> None:
        _, results = run_named_benchmarks(
            op_names=["addmm"],
            device_name="cpu",
            dtype_name="float32",
            min_run_time=0.01,
        )
        rendered = json.dumps([result.to_dict() for result in results])
        self.assertIn("addmm", rendered)


if __name__ == "__main__":
    unittest.main()
