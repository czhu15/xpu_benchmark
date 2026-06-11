from __future__ import annotations

import statistics
import timeit
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.utils.benchmark as benchmark

from xpu_benchmark.ops import BENCHMARK_SPECS


TimerBackend = Literal["torch", "timeit", "event"]
TIMER_BACKENDS: tuple[TimerBackend, ...] = ("torch", "timeit", "event")


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
}

DEFAULT_PEAK_TFLOPS_BF16 = 183.0
DEFAULT_PEAK_BANDWIDTH_GBPS = 608.0


@dataclass(frozen=True)
class BenchmarkResult:
    op_name: str
    label: str
    description: str
    input_shape: str
    dtype_name: str
    timer_backend: str
    env: str
    params: dict[str, Any]
    median_seconds: float
    mean_seconds: float
    number_per_run: int
    estimated_flops: float
    estimated_bytes: float
    arithmetic_intensity: float
    tflops: float
    bandwidth_gbps: float
    peak_tflops_percent: float
    peak_bandwidth_percent: float
    roofline_tflops: float
    eff_vs_roofline_percent: float
    bound_hint: str
    peak_tflops: float
    peak_bandwidth_gbps: float

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
    is_backward = op_name.endswith("_backward")
    base_op_name = op_name.removesuffix("_backward") if is_backward else op_name

    def format_direction(shape: str) -> str:
        return f"backward: {shape}" if is_backward else shape

    if base_op_name == "addmm":
        return format_direction(
            f"mat1=({params['m']}, {params['k']}), "
            f"mat2=({params['k']}, {params['n']}), "
            f"bias=({params['m']}, {params['n']})"
        )
    if base_op_name == "bmm":
        return format_direction(
            f"lhs=({params['batch']}, {params['m']}, {params['k']}), "
            f"rhs=({params['batch']}, {params['k']}, {params['n']})"
        )
    if base_op_name == "group_gemm":
        return format_direction(
            f"{params['groups']} groups: lhs=({params['m']}, {params['k']}), rhs=({params['k']}, {params['n']})"
        )
    if base_op_name == "layernorm":
        return format_direction(f"x=({params['batch']}, {params['sequence']}, {params['hidden']})")
    if base_op_name == "sum":
        return format_direction(f"x=({params['d0']}, {params['d1']}, {params['d2']}), dim={params['dim']}")
    if base_op_name == "concat":
        return format_direction(f"{params['count']} tensors: ({params['rows']}, {params['cols']}), dim={params['dim']}")
    if base_op_name == "copy":
        return format_direction(f"src/dst=({params['rows']}, {params['cols']})")
    if base_op_name == "fused_attention_score":
        shape = f"({params['batch']}, {params['heads']}, {params['sequence']}, {params['head_dim']})"
        return format_direction(f"q/k/v={shape}")
    if base_op_name == "triton_varlen_flash_attention":
        if "total_q" in params:
            return format_direction(
                f"q=({params['total_q']}, {params['num_heads']}, {params['head_dim']}), "
                f"k/v=({params['total_k']}, {params['num_heads']}, {params['head_dim']}), "
                f"batch={params['batch']}, max_q={params['max_seqlen_q']}, max_k={params['max_seqlen_k']}"
            )
        shape = f"({params['sequence']}, {params['heads']}, {params['head_dim']})"
        return format_direction(f"q/k/v={shape}")
    if base_op_name == "triton_swiglu":
        return format_direction(
            f"x=({params['tokens']}, {params['hidden']}), intermediate={params['intermediate']}"
        )
    return format_direction(_format_params(params))


def _sum_output_elements(params: dict[str, Any]) -> int:
    dims = [int(params["d0"]), int(params["d1"]), int(params["d2"])]
    dim = int(params["dim"])
    if dim < 0:
        dim += len(dims)
    output_elements = 1
    for index, size in enumerate(dims):
        if index != dim:
            output_elements *= size
    return output_elements


def _dense_attention_work(batch: int, heads: int, sequence: int, head_dim: int, dtype_size: int) -> tuple[float, float]:
    flops = 4.0 * batch * heads * sequence * sequence * head_dim
    bytes_accessed = float(4 * batch * heads * sequence * head_dim * dtype_size)
    return flops, bytes_accessed


def _varlen_attention_work(params: dict[str, Any], dtype_size: int) -> tuple[float, float]:
    batch = int(params["batch"])
    heads = int(params["num_heads"])
    head_dim = int(params["head_dim"])
    total_q = int(params["total_q"])
    total_k = int(params["total_k"])
    average_q = total_q / batch
    average_k = total_k / batch
    qk_pairs = batch * average_q * average_k
    flops = 4.0 * heads * qk_pairs * head_dim
    bytes_accessed = float((2 * total_q + 2 * total_k) * heads * head_dim * dtype_size)
    bytes_accessed += float((total_q + total_k + 2 * (batch + 1)) * 4)
    return flops, bytes_accessed


def _estimate_workload(op_name: str, params: dict[str, Any], dtype_name: str) -> tuple[float, float]:
    dtype_size = DTYPE_BYTES[dtype_name]
    is_backward = op_name.endswith("_backward")
    base_op_name = op_name.removesuffix("_backward") if is_backward else op_name
    backward_multiplier = 2.0 if is_backward else 1.0

    if base_op_name == "addmm":
        m, k, n = int(params["m"]), int(params["k"]), int(params["n"])
        flops = 2.0 * m * n * k + m * n
        bytes_accessed = float((m * k + k * n + 2 * m * n) * dtype_size)
    elif base_op_name == "bmm":
        batch, m, k, n = int(params["batch"]), int(params["m"]), int(params["k"]), int(params["n"])
        flops = 2.0 * batch * m * n * k
        bytes_accessed = float(batch * (m * k + k * n + m * n) * dtype_size)
    elif base_op_name == "group_gemm":
        groups, m, k, n = int(params["groups"]), int(params["m"]), int(params["k"]), int(params["n"])
        flops = 2.0 * groups * m * n * k
        bytes_accessed = float(groups * (m * k + k * n + m * n) * dtype_size)
    elif base_op_name == "layernorm":
        elements = int(params["batch"]) * int(params["sequence"]) * int(params["hidden"])
        flops = 5.0 * elements
        bytes_accessed = float(2 * elements * dtype_size)
    elif base_op_name == "sum":
        input_elements = int(params["d0"]) * int(params["d1"]) * int(params["d2"])
        output_elements = _sum_output_elements(params)
        flops = float(max(input_elements - output_elements, 0))
        bytes_accessed = float((input_elements + output_elements) * dtype_size)
    elif base_op_name == "concat":
        elements = int(params["count"]) * int(params["rows"]) * int(params["cols"])
        flops = 0.0
        bytes_accessed = float(2 * elements * dtype_size)
    elif base_op_name == "copy":
        elements = int(params["rows"]) * int(params["cols"])
        flops = 0.0
        bytes_accessed = float(2 * elements * dtype_size)
    elif base_op_name == "fused_attention_score":
        flops, bytes_accessed = _dense_attention_work(
            int(params["batch"]),
            int(params["heads"]),
            int(params["sequence"]),
            int(params["head_dim"]),
            dtype_size,
        )
    elif base_op_name == "triton_varlen_flash_attention":
        flops, bytes_accessed = _varlen_attention_work(params, dtype_size)
    elif base_op_name == "triton_swiglu":
        tokens = int(params["tokens"])
        hidden = int(params["hidden"])
        intermediate = int(params["intermediate"])
        flops = 4.0 * tokens * hidden * intermediate + 4.0 * tokens * intermediate
        bytes_accessed = float((tokens * hidden + 2 * hidden * intermediate + 4 * intermediate + tokens * intermediate) * dtype_size)
    else:
        flops = 0.0
        bytes_accessed = 0.0

    return flops * backward_multiplier, bytes_accessed * backward_multiplier


def _roofline_metrics(
    op_name: str,
    params: dict[str, Any],
    dtype_name: str,
    median_seconds: float,
    peak_tflops: float,
    peak_bandwidth_gbps: float,
) -> dict[str, float | str]:
    estimated_flops, estimated_bytes = _estimate_workload(op_name, params, dtype_name)
    tflops = estimated_flops / median_seconds / 1.0e12 if median_seconds > 0 else 0.0
    bandwidth_gbps = estimated_bytes / median_seconds / 1.0e9 if median_seconds > 0 else 0.0
    arithmetic_intensity = tflops * 1000.0 / bandwidth_gbps if bandwidth_gbps > 0 else 0.0
    peak_tflops_percent = tflops / peak_tflops * 100.0 if peak_tflops > 0 else 0.0
    peak_bandwidth_percent = bandwidth_gbps / peak_bandwidth_gbps * 100.0 if peak_bandwidth_gbps > 0 else 0.0
    memory_limited_tflops = arithmetic_intensity * peak_bandwidth_gbps / 1000.0
    roofline_tflops = min(peak_tflops, memory_limited_tflops) if arithmetic_intensity > 0 else 0.0
    eff_vs_roofline_percent = tflops / roofline_tflops * 100.0 if roofline_tflops > 0 else 0.0
    ridge_point = peak_tflops * 1000.0 / peak_bandwidth_gbps if peak_bandwidth_gbps > 0 else float("inf")
    bound_hint = "memory-bound" if arithmetic_intensity < ridge_point else "compute-bound"
    return {
        "estimated_flops": estimated_flops,
        "estimated_bytes": estimated_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "tflops": tflops,
        "bandwidth_gbps": bandwidth_gbps,
        "peak_tflops_percent": peak_tflops_percent,
        "peak_bandwidth_percent": peak_bandwidth_percent,
        "roofline_tflops": roofline_tflops,
        "eff_vs_roofline_percent": eff_vs_roofline_percent,
        "bound_hint": bound_hint,
        "peak_tflops": peak_tflops,
        "peak_bandwidth_gbps": peak_bandwidth_gbps,
    }


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


def _event_benchmark(
    runner: Any,
    device: torch.device,
    warmup: int = 10,
    repeat: int = 100,
) -> tuple[float, float]:
    import xpu_benchmark.ops as ops_module

    for _ in range(warmup):
        runner()

    ops_module._sync_enabled = False
    try:
        if device.type == "xpu":
            starts = [torch.xpu.Event(enable_timing=True) for _ in range(repeat)]
            ends = [torch.xpu.Event(enable_timing=True) for _ in range(repeat)]
        elif device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        else:
            raise RuntimeError(f"Event timing is not supported for device type '{device.type}'.")

        for index in range(repeat):
            starts[index].record()
            runner()
            ends[index].record()
        ends[-1].synchronize()
        elapsed_ms = [starts[index].elapsed_time(ends[index]) for index in range(repeat)]
    finally:
        ops_module._sync_enabled = True

    elapsed_seconds = [elapsed / 1000.0 for elapsed in elapsed_ms]
    return statistics.median(elapsed_seconds), statistics.fmean(elapsed_seconds)


def run_named_benchmarks(
    op_names: list[str],
    device_name: str,
    dtype_name: str,
    min_run_time: float,
    timer_backend: TimerBackend = "torch",
    runs: int | None = None,
    peak_tflops: float = DEFAULT_PEAK_TFLOPS_BF16,
    peak_bandwidth_gbps: float = DEFAULT_PEAK_BANDWIDTH_GBPS,
) -> tuple[list[benchmark.Measurement], list[BenchmarkResult]]:
    if timer_backend not in TIMER_BACKENDS:
        supported = ", ".join(TIMER_BACKENDS)
        raise ValueError(f"Unsupported timer backend '{timer_backend}'. Choose from: {supported}.")
    if runs is not None and runs < 1:
        raise ValueError("runs must be a positive integer when provided.")
    if peak_tflops <= 0:
        raise ValueError("peak_tflops must be positive.")
    if peak_bandwidth_gbps <= 0:
        raise ValueError("peak_bandwidth_gbps must be positive.")

    device = torch.device(device_name)
    validate_device(device)
    dtype = resolve_dtype(dtype_name)
    env = f"device={device.type}, dtype={dtype_name}, timer={timer_backend}"

    measurements: list[benchmark.Measurement] = []
    results: list[BenchmarkResult] = []

    for op_name in op_names:
        spec = BENCHMARK_SPECS[op_name]
        for params in spec.cases:
            runner = spec.build(device, dtype, params)
            if timer_backend == "torch":
                timer = benchmark.Timer(
                    stmt="benchmark_fn()",
                    globals={"benchmark_fn": runner},
                    label=spec.label,
                    sub_label=op_name,
                    description=_format_params(params),
                    env=env,
                    num_threads=1,
                )
                measurement = timer.timeit(runs) if runs is not None else timer.blocked_autorange(min_run_time=min_run_time)
                measurements.append(measurement)
                median_seconds = measurement.median
                mean_seconds = measurement.mean
                number_per_run = measurement.number_per_run
            elif timer_backend == "event":
                number_per_run = runs if runs is not None else 100
                median_seconds, mean_seconds = _event_benchmark(runner, device, repeat=number_per_run)
            else:
                timer = timeit.Timer(
                    stmt="benchmark_fn()",
                    globals={"benchmark_fn": runner},
                )
                timer.timeit(number=1)
                number_per_run = runs if runs is not None else _timeit_autorange(timer, min_run_time=min_run_time)
                per_run_seconds = [
                    timer.timeit(number=number_per_run) / number_per_run
                    for _ in range(5)
                ]
                median_seconds = statistics.median(per_run_seconds)
                mean_seconds = statistics.fmean(per_run_seconds)
            roofline = _roofline_metrics(
                op_name=op_name,
                params=params,
                dtype_name=dtype_name,
                median_seconds=median_seconds,
                peak_tflops=peak_tflops,
                peak_bandwidth_gbps=peak_bandwidth_gbps,
            )
            results.append(
                BenchmarkResult(
                    op_name=op_name,
                    label=spec.label,
                    description=spec.description,
                    input_shape=_format_input_shape(op_name, params),
                    dtype_name=dtype_name,
                    timer_backend=timer_backend,
                    env=env,
                    params=params,
                    median_seconds=median_seconds,
                    mean_seconds=mean_seconds,
                    number_per_run=number_per_run,
                    **roofline,
                )
            )

    return measurements, results
