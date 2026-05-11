from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

BenchmarkRunner = Callable[[], Any]
BenchmarkBuilder = Callable[[torch.device, torch.dtype], BenchmarkRunner]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    label: str
    description: str
    params: dict[str, Any]
    build: BenchmarkBuilder


def _synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _addmm_spec() -> BenchmarkSpec:
    params = {"m": 512, "k": 512, "n": 512}

    def build(device: torch.device, dtype: torch.dtype) -> BenchmarkRunner:
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
        params=params,
        build=build,
    )


def _bmm_spec() -> BenchmarkSpec:
    params = {"batch": 16, "m": 128, "k": 128, "n": 128}

    def build(device: torch.device, dtype: torch.dtype) -> BenchmarkRunner:
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
        params=params,
        build=build,
    )


def _group_gemm_spec() -> BenchmarkSpec:
    params = {"groups": 4, "m": 256, "k": 256, "n": 256}

    def build(device: torch.device, dtype: torch.dtype) -> BenchmarkRunner:
        lhs = [
            torch.randn((params["m"], params["k"]), device=device, dtype=dtype)
            for _ in range(params["groups"])
        ]
        rhs = [
            torch.randn((params["k"], params["n"]), device=device, dtype=dtype)
            for _ in range(params["groups"])
        ]

        def run() -> torch.Tensor:
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
        params=params,
        build=build,
    )


def _layernorm_spec() -> BenchmarkSpec:
    params = {"batch": 32, "sequence": 256, "hidden": 768}

    def build(device: torch.device, dtype: torch.dtype) -> BenchmarkRunner:
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
        params=params,
        build=build,
    )


def _sum_spec() -> BenchmarkSpec:
    params = {"d0": 128, "d1": 256, "d2": 512, "dim": -1}

    def build(device: torch.device, dtype: torch.dtype) -> BenchmarkRunner:
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
        params=params,
        build=build,
    )


def _concat_spec() -> BenchmarkSpec:
    params = {"count": 4, "rows": 512, "cols": 256, "dim": 1}

    def build(device: torch.device, dtype: torch.dtype) -> BenchmarkRunner:
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
        params=params,
        build=build,
    )


def _copy_spec() -> BenchmarkSpec:
    params = {"rows": 2048, "cols": 1024}

    def build(device: torch.device, dtype: torch.dtype) -> BenchmarkRunner:
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
        params=params,
        build=build,
    )


def _fused_attention_score_spec() -> BenchmarkSpec:
    params = {"batch": 4, "heads": 16, "sequence": 128, "head_dim": 64}

    def build(device: torch.device, dtype: torch.dtype) -> BenchmarkRunner:
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
        params=params,
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
