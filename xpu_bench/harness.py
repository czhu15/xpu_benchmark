from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.utils.benchmark as benchmark

from xpu_bench.ops import BENCHMARK_SPECS


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


def run_named_benchmarks(
    op_names: list[str],
    device_name: str,
    dtype_name: str,
    min_run_time: float,
) -> tuple[list[benchmark.Measurement], list[BenchmarkResult]]:
    device = torch.device(device_name)
    validate_device(device)
    dtype = resolve_dtype(dtype_name)
    env = f"device={device.type}, dtype={dtype_name}"

    measurements: list[benchmark.Measurement] = []
    results: list[BenchmarkResult] = []

    for op_name in op_names:
        spec = BENCHMARK_SPECS[op_name]
        runner = spec.build(device, dtype)
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
        results.append(
            BenchmarkResult(
                op_name=op_name,
                label=spec.label,
                description=spec.description,
                env=env,
                params=spec.params,
                median_seconds=measurement.median,
                mean_seconds=measurement.mean,
                iqr_seconds=measurement.iqr,
                number_per_run=measurement.number_per_run,
            )
        )

    return measurements, results
