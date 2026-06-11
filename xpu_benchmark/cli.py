from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch.utils.benchmark as benchmark

from xpu_benchmark.harness import (
    DEFAULT_PEAK_BANDWIDTH_GBPS,
    DEFAULT_PEAK_TFLOPS_BF16,
    DTYPE_MAP,
    TIMER_BACKENDS,
    run_named_benchmarks,
)
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
        default="bfloat16",
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
        "--runs",
        type=int,
        help="Fixed number of benchmark iterations per measurement. Overrides --min-run-time autoranging.",
    )
    run_parser.add_argument(
        "--timer",
        choices=TIMER_BACKENDS,
        default="timeit",
        help=(
            "Timer backend to use: event uses accelerator events for device-side kernel time; "
            "torch uses torch.utils.benchmark.Timer; timeit uses Python timeit.Timer."
        ),
    )
    run_parser.add_argument(
        "--peak-tflops",
        type=float,
        default=DEFAULT_PEAK_TFLOPS_BF16,
        help="Theoretical peak compute throughput in TFLOPS used for roofline percentages. Default: 183 for BF16.",
    )
    run_parser.add_argument(
        "--peak-bandwidth-gbps",
        type=float,
        default=DEFAULT_PEAK_BANDWIDTH_GBPS,
        help="Theoretical peak bandwidth in GB/s used for roofline percentages. Default: 608.",
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
    op_width = max([len("op"), *(len(str(result["op_name"])) for result in results)])
    shape_width = max([len("input shape"), *(len(str(result["input_shape"])) for result in results)])
    dtype_width = max([len("dtype"), *(len(str(result.get("dtype_name", ""))) for result in results)])
    lines = [
        f"{'op':>{op_width}}  {'input shape':<{shape_width}}  {'dtype':>{dtype_width}}  {'timer':>6}  {'median (us)':>12}  {'mean (us)':>10}  {'runs':>8}  {'TFLOPS':>9}  {'Bandwidth (GB/s)':>16}  {'%peak_tflops':>12}  {'%peak_bw':>8}  {'AI(F/B)':>8}  {'roofline_tflops':>16}  {'eff_vs_roofline(%)':>19}  {'bound_hint':>15}",
    ]
    lines.append("-" * len(lines[0]))
    for result in results:
        lines.append(
            f"{str(result['op_name']):>{op_width}}  "
            f"{str(result['input_shape']):<{shape_width}}  "
            f"{str(result.get('dtype_name', '')):>{dtype_width}}  "
            f"{str(result['timer_backend']):>6}  "
            f"{float(result['median_seconds']) * 1e6:12.2f}  "
            f"{float(result['mean_seconds']) * 1e6:10.2f}  "
            f"{int(result['number_per_run']):8d}  "
            f"{float(result.get('tflops', 0.0)):9.2f}  "
            f"{float(result.get('bandwidth_gbps', 0.0)):16.2f}  "
            f"{float(result.get('peak_tflops_percent', 0.0)):12.2f}  "
            f"{float(result.get('peak_bandwidth_percent', 0.0)):8.2f}  "
            f"{float(result.get('arithmetic_intensity', 0.0)):8.2f}  "
            f"{float(result.get('roofline_tflops', 0.0)):16.2f}  "
            f"{float(result.get('eff_vs_roofline_percent', 0.0)):19.2f}  "
            f"{str(result.get('bound_hint', '')):>15}"
        )
    return "\n".join(lines)


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
        runs=args.runs,
        peak_tflops=args.peak_tflops,
        peak_bandwidth_gbps=args.peak_bandwidth_gbps,
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
