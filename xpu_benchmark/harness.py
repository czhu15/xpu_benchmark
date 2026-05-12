from __future__ import annotations

import statistics
import timeit
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.utils.benchmark as benchmark

from xpu_benchmark.ops import BENCHMARK_SPECS


TimerBackend = Literal["torch", "timeit"]
TIMER_BACKENDS: tuple[TimerBackend, ...] = ("torch", "timeit")


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class BenchmarkResult:
    op_name: str
    label: str
    description: str
    input_shape: str
    timer_backend: str
    env: str
    params: dict[str, Any]
    median_seconds: float
    mean_seconds: float
    iqr_seconds: float
    number_per_run: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return DTYPE_MAP[dtype_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_MAP))
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Choose from: {supported}.") from exc


def validate_device(device: torch.device) -> None:
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU device requested, but torch.xpu.is_available() is false.")


def _format_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _format_input_shape(op_name: str, params: dict[str, Any]) -> str:
    if op_name == "addmm":
        return (
            f"mat1=({params['m']}, {params['k']}), "
            f"mat2=({params['k']}, {params['n']}), "
            f"bias=({params['m']}, {params['n']})"
        )
    if op_name == "bmm":
        return (
            f"lhs=({params['batch']}, {params['m']}, {params['k']}), "
            f"rhs=({params['batch']}, {params['k']}, {params['n']})"
        )
    if op_name == "group_gemm":
        return f"{params['groups']} groups: lhs=({params['m']}, {params['k']}), rhs=({params['k']}, {params['n']})"
    if op_name == "layernorm":
        return f"x=({params['batch']}, {params['sequence']}, {params['hidden']})"
    if op_name == "sum":
        return f"x=({params['d0']}, {params['d1']}, {params['d2']}), dim={params['dim']}"
    if op_name == "concat":
        return f"{params['count']} tensors: ({params['rows']}, {params['cols']}), dim={params['dim']}"
    if op_name == "copy":
        return f"src/dst=({params['rows']}, {params['cols']})"
    if op_name == "fused_attention_score":
        shape = f"({params['batch']}, {params['heads']}, {params['sequence']}, {params['head_dim']})"
        return f"q/k/v={shape}"
    return _format_params(params)


def _iqr(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def _timeit_autorange(timer: timeit.Timer, min_run_time: float) -> int:
    target_run_time = max(min_run_time, 0.0)
    scale = 1
    while True:
        for multiplier in (1, 2, 5):
            number = multiplier * scale
            total_seconds = timer.timeit(number=number)
            if total_seconds >= target_run_time:
                return number
        scale *= 10


def run_named_benchmarks(
    op_names: list[str],
    device_name: str,
    dtype_name: str,
    min_run_time: float,
    timer_backend: TimerBackend = "torch",
) -> tuple[list[benchmark.Measurement], list[BenchmarkResult]]:
    if timer_backend not in TIMER_BACKENDS:
        supported = ", ".join(TIMER_BACKENDS)
        raise ValueError(f"Unsupported timer backend '{timer_backend}'. Choose from: {supported}.")

    device = torch.device(device_name)
    validate_device(device)
    dtype = resolve_dtype(dtype_name)
    env = f"device={device.type}, dtype={dtype_name}, timer={timer_backend}"

    measurements: list[benchmark.Measurement] = []
    results: list[BenchmarkResult] = []

    for op_name in op_names:
        spec = BENCHMARK_SPECS[op_name]
        runner = spec.build(device, dtype)
        if timer_backend == "torch":
            timer = benchmark.Timer(
                stmt="benchmark_fn()",
                globals={"benchmark_fn": runner},
                label=spec.label,
                sub_label=op_name,
                description=_format_params(spec.params),
                env=env,
                num_threads=1,
            )
            measurement = timer.blocked_autorange(min_run_time=min_run_time)
            measurements.append(measurement)
            median_seconds = measurement.median
            mean_seconds = measurement.mean
            iqr_seconds = measurement.iqr
            number_per_run = measurement.number_per_run
        else:
            timer = timeit.Timer(
                stmt="benchmark_fn()",
                globals={"benchmark_fn": runner},
            )
            timer.timeit(number=1)
            number_per_run = _timeit_autorange(timer, min_run_time=min_run_time)
            per_run_seconds = [
                timer.timeit(number=number_per_run) / number_per_run
                for _ in range(5)
            ]
            median_seconds = statistics.median(per_run_seconds)
            mean_seconds = statistics.fmean(per_run_seconds)
            iqr_seconds = _iqr(per_run_seconds)
        results.append(
            BenchmarkResult(
                op_name=op_name,
                label=spec.label,
                description=spec.description,
                input_shape=_format_input_shape(op_name, spec.params),
                timer_backend=timer_backend,
                env=env,
                params=spec.params,
                median_seconds=median_seconds,
                mean_seconds=mean_seconds,
                iqr_seconds=iqr_seconds,
                number_per_run=number_per_run,
            )
        )

    return measurements, results
