from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch.utils.benchmark as benchmark

from xpu_benchmark.harness import DTYPE_MAP, TIMER_BACKENDS, run_named_benchmarks
from xpu_benchmark.ops import BENCHMARK_SPECS, list_benchmarks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark selected PyTorch ops on Intel XPU.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List the available benchmark cases.")
    list_parser.set_defaults(command="list")

    run_parser = subparsers.add_parser("run", help="Run one or more benchmark cases.")
    run_parser.add_argument(
        "--ops",
        nargs="+",
        default=sorted(BENCHMARK_SPECS),
        choices=sorted(BENCHMARK_SPECS),
        help="Benchmark cases to run.",
    )
    run_parser.add_argument("--device", default="xpu", help="Device to benchmark on, for example xpu or cpu.")
    run_parser.add_argument(
        "--dtype",
        default="float32",
        choices=sorted(DTYPE_MAP),
        help="Tensor dtype to use for benchmark inputs.",
    )
    run_parser.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        help="Minimum runtime in seconds used to choose the number of iterations per benchmark.",
    )
    run_parser.add_argument(
        "--timer",
        choices=TIMER_BACKENDS,
        default="torch",
        help="Timer backend to use: torch uses torch.utils.benchmark.Timer; timeit uses Python timeit.Timer.",
    )
    run_parser.add_argument(
        "--format",
        choices=("compare", "json"),
        default="compare",
        help="Output format.",
    )
    run_parser.add_argument("--output", help="Optional output file path.")
    run_parser.set_defaults(command="run")
    return parser


def _render_list() -> str:
    lines = []
    for spec in list_benchmarks():
        lines.append(f"{spec.name:>22}  {spec.description}")
    return "\n".join(lines)


def _render_results(
    measurements: list[benchmark.Measurement],
    results: list[dict[str, object]],
    output_format: str,
) -> str:
    if output_format == "json":
        return json.dumps(results, indent=2)
    if not measurements:
        lines = [
            f"{'op':>22}  {'timer':>6}  {'median (us)':>12}  {'mean (us)':>10}  {'iqr (us)':>9}  {'runs':>8}",
            "-" * 77,
        ]
        for result in results:
            lines.append(
                f"{str(result['op_name']):>22}  "
                f"{str(result['timer_backend']):>6}  "
                f"{float(result['median_seconds']) * 1e6:12.2f}  "
                f"{float(result['mean_seconds']) * 1e6:10.2f}  "
                f"{float(result['iqr_seconds']) * 1e6:9.2f}  "
                f"{int(result['number_per_run']):8d}"
            )
        return "\n".join(lines)
    comparison = benchmark.Compare(measurements)
    comparison.trim_significant_figures()
    return str(comparison)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        print(_render_list())
        return 0

    measurements, result_objects = run_named_benchmarks(
        op_names=list(args.ops),
        device_name=args.device,
        dtype_name=args.dtype,
        min_run_time=args.min_run_time,
        timer_backend=args.timer,
    )
    rendered = _render_results(
        measurements=measurements,
        results=[result.to_dict() for result in result_objects],
        output_format=args.format,
    )
    print(rendered)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{rendered}\n", encoding="utf-8")

    return 0
