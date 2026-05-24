from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

BenchmarkRunner = Callable[[], Any]
BenchmarkBuilder = Callable[[torch.device, torch.dtype, dict[str, Any]], BenchmarkRunner]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    label: str
    description: str
    cases: list[dict[str, Any]]
    build: BenchmarkBuilder


def _load_benchmark_cases() -> dict[str, list[dict[str, Any]]]:
    config_path = Path(__file__).with_name("benchmark_shapes.json")
    raw_cases = json.loads(config_path.read_text(encoding="utf-8"))

    benchmark_cases: dict[str, list[dict[str, Any]]] = {}
    for op_name, cases in raw_cases.items():
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"Benchmark shape config for '{op_name}' must be a non-empty list.")
        benchmark_cases[op_name] = []
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError(f"Each benchmark shape config for '{op_name}' must be an object.")
            benchmark_cases[op_name].append(dict(case))
    return benchmark_cases


BENCHMARK_CASES = _load_benchmark_cases()


def _cases_for(op_name: str) -> list[dict[str, Any]]:
    try:
        return BENCHMARK_CASES[op_name]
    except KeyError as exc:
        raise ValueError(f"Missing benchmark shape config for '{op_name}'.") from exc


def _synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _addmm_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        mat1 = torch.randn((params["m"], params["k"]), device=device, dtype=dtype)
        mat2 = torch.randn((params["k"], params["n"]), device=device, dtype=dtype)
        bias = torch.randn((params["m"], params["n"]), device=device, dtype=dtype)

        def run() -> torch.Tensor:
            result = torch.addmm(bias, mat1, mat2)
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="addmm",
        label="Matmul",
        description="torch.addmm",
        cases=_cases_for("addmm"),
        build=build,
    )


def _bmm_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        lhs = torch.randn((params["batch"], params["m"], params["k"]), device=device, dtype=dtype)
        rhs = torch.randn((params["batch"], params["k"], params["n"]), device=device, dtype=dtype)

        def run() -> torch.Tensor:
            result = torch.bmm(lhs, rhs)
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="bmm",
        label="Batched matmul",
        description="torch.bmm",
        cases=_cases_for("bmm"),
        build=build,
    )


def _group_gemm_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        lhs = [
            torch.randn((params["m"], params["k"]), device=device, dtype=dtype)
            for _ in range(params["groups"])
        ]
        rhs = [
            torch.randn((params["k"], params["n"]), device=device, dtype=dtype)
            for _ in range(params["groups"])
        ]

        def run() -> torch.Tensor:
            # Current implementation of grouped GEMM is to loop over groups on each GEMM,
            # This is not the most efficient way to do grouped GEMM. Need call the specific 
            # grouped GEMM kernel when it is available.
            result = torch.empty((params["m"], params["n"]), device=device, dtype=dtype)
            for left, right in zip(lhs, rhs):
                result = torch.matmul(left, right)
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="group_gemm",
        label="Grouped GEMM",
        description="grouped torch.matmul workload",
        cases=_cases_for("group_gemm"),
        build=build,
    )


def _layernorm_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        x = torch.randn((params["batch"], params["sequence"], params["hidden"]), device=device, dtype=dtype)
        weight = torch.randn((params["hidden"],), device=device, dtype=dtype)
        bias = torch.randn((params["hidden"],), device=device, dtype=dtype)

        def run() -> torch.Tensor:
            result = F.layer_norm(x, (params["hidden"],), weight, bias)
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="layernorm",
        label="LayerNorm",
        description="torch.nn.functional.layer_norm",
        cases=_cases_for("layernorm"),
        build=build,
    )


def _sum_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        x = torch.randn((params["d0"], params["d1"], params["d2"]), device=device, dtype=dtype)

        def run() -> torch.Tensor:
            result = torch.sum(x, dim=params["dim"])
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="sum",
        label="Reduction",
        description="torch.sum",
        cases=_cases_for("sum"),
        build=build,
    )


def _concat_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        tensors = [
            torch.randn((params["rows"], params["cols"]), device=device, dtype=dtype)
            for _ in range(params["count"])
        ]

        def run() -> torch.Tensor:
            result = torch.cat(tensors, dim=params["dim"])
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="concat",
        label="Concat",
        description="torch.cat",
        cases=_cases_for("concat"),
        build=build,
    )


def _copy_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        src = torch.randn((params["rows"], params["cols"]), device=device, dtype=dtype)
        dst = torch.empty_like(src)

        def run() -> torch.Tensor:
            result = dst.copy_(src)
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="copy",
        label="Copy",
        description="Tensor.copy_",
        cases=_cases_for("copy"),
        build=build,
    )


def _fused_attention_score_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        query = torch.randn(
            (params["batch"], params["heads"], params["sequence"], params["head_dim"]),
            device=device,
            dtype=dtype,
        )
        key = torch.randn(
            (params["batch"], params["heads"], params["sequence"], params["head_dim"]),
            device=device,
            dtype=dtype,
        )
        value = torch.randn(
            (params["batch"], params["heads"], params["sequence"], params["head_dim"]),
            device=device,
            dtype=dtype,
        )

        def run() -> torch.Tensor:
            result = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="fused_attention_score",
        label="Fused attention",
        description="torch.nn.functional.scaled_dot_product_attention",
        cases=_cases_for("fused_attention_score"),
        build=build,
    )


BENCHMARK_SPECS = {
    spec.name: spec
    for spec in (
        _addmm_spec(),
        _bmm_spec(),
        _group_gemm_spec(),
        _layernorm_spec(),
        _sum_spec(),
        _concat_spec(),
        _copy_spec(),
        _fused_attention_score_spec(),
    )
}


def list_benchmarks() -> list[BenchmarkSpec]:
    return [BENCHMARK_SPECS[name] for name in sorted(BENCHMARK_SPECS)]
