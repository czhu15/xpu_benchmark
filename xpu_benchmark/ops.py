from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from xpu_benchmark.triton_flash_attention import triton_flash_attention
from xpu_benchmark.triton_swiglu import triton_swiglu

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


def _clear_gradients(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


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


def _addmm_backward_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        mat1 = torch.randn((params["m"], params["k"]), device=device, dtype=dtype, requires_grad=True)
        mat2 = torch.randn((params["k"], params["n"]), device=device, dtype=dtype, requires_grad=True)
        bias = torch.randn((params["m"], params["n"]), device=device, dtype=dtype, requires_grad=True)
        grad_output = torch.ones((params["m"], params["n"]), device=device, dtype=dtype)

        def run() -> torch.Tensor | None:
            _clear_gradients(mat1, mat2, bias)
            result = torch.addmm(bias, mat1, mat2)
            result.backward(grad_output)
            _synchronize(device)
            return mat1.grad

        return run

    return BenchmarkSpec(
        name="addmm_backward",
        label="Matmul backward",
        description="torch.addmm backward",
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


def _bmm_backward_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        lhs = torch.randn((params["batch"], params["m"], params["k"]), device=device, dtype=dtype, requires_grad=True)
        rhs = torch.randn((params["batch"], params["k"], params["n"]), device=device, dtype=dtype, requires_grad=True)
        grad_output = torch.ones((params["batch"], params["m"], params["n"]), device=device, dtype=dtype)

        def run() -> torch.Tensor | None:
            _clear_gradients(lhs, rhs)
            result = torch.bmm(lhs, rhs)
            result.backward(grad_output)
            _synchronize(device)
            return lhs.grad

        return run

    return BenchmarkSpec(
        name="bmm_backward",
        label="Batched matmul backward",
        description="torch.bmm backward",
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


def _group_gemm_backward_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        lhs = [
            torch.randn((params["m"], params["k"]), device=device, dtype=dtype, requires_grad=True)
            for _ in range(params["groups"])
        ]
        rhs = [
            torch.randn((params["k"], params["n"]), device=device, dtype=dtype, requires_grad=True)
            for _ in range(params["groups"])
        ]

        def run() -> torch.Tensor | None:
            for tensor in (*lhs, *rhs):
                tensor.grad = None
            total = torch.zeros((), device=device, dtype=dtype)
            for left, right in zip(lhs, rhs):
                total = total + torch.matmul(left, right).sum()
            total.backward()
            _synchronize(device)
            return lhs[0].grad

        return run

    return BenchmarkSpec(
        name="group_gemm_backward",
        label="Grouped GEMM backward",
        description="grouped torch.matmul workload backward",
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


def _layernorm_backward_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        x = torch.randn(
            (params["batch"], params["sequence"], params["hidden"]),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        weight = torch.randn((params["hidden"],), device=device, dtype=dtype, requires_grad=True)
        bias = torch.randn((params["hidden"],), device=device, dtype=dtype, requires_grad=True)
        grad_output = torch.ones_like(x)

        def run() -> torch.Tensor | None:
            _clear_gradients(x, weight, bias)
            result = F.layer_norm(x, (params["hidden"],), weight, bias)
            result.backward(grad_output)
            _synchronize(device)
            return x.grad

        return run

    return BenchmarkSpec(
        name="layernorm_backward",
        label="LayerNorm backward",
        description="torch.nn.functional.layer_norm backward",
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


def _sum_backward_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        x = torch.randn((params["d0"], params["d1"], params["d2"]), device=device, dtype=dtype, requires_grad=True)
        output_shape = list(x.shape)
        del output_shape[params["dim"]]
        grad_output = torch.ones(tuple(output_shape), device=device, dtype=dtype)

        def run() -> torch.Tensor | None:
            _clear_gradients(x)
            result = torch.sum(x, dim=params["dim"])
            result.backward(grad_output)
            _synchronize(device)
            return x.grad

        return run

    return BenchmarkSpec(
        name="sum_backward",
        label="Reduction backward",
        description="torch.sum backward",
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


def _concat_backward_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        tensors = [
            torch.randn((params["rows"], params["cols"]), device=device, dtype=dtype, requires_grad=True)
            for _ in range(params["count"])
        ]
        output_shape = [params["rows"], params["cols"]]
        output_shape[params["dim"]] *= params["count"]
        grad_output = torch.ones(tuple(output_shape), device=device, dtype=dtype)

        def run() -> torch.Tensor | None:
            for tensor in tensors:
                tensor.grad = None
            result = torch.cat(tensors, dim=params["dim"])
            result.backward(grad_output)
            _synchronize(device)
            return tensors[0].grad

        return run

    return BenchmarkSpec(
        name="concat_backward",
        label="Concat backward",
        description="torch.cat backward",
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


def _copy_backward_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        src = torch.randn((params["rows"], params["cols"]), device=device, dtype=dtype, requires_grad=True)
        base_dst = torch.empty_like(src)
        grad_output = torch.ones_like(src)

        def run() -> torch.Tensor | None:
            _clear_gradients(src)
            dst = base_dst.detach()
            result = dst.copy_(src)
            result.backward(grad_output)
            _synchronize(device)
            return src.grad

        return run

    return BenchmarkSpec(
        name="copy_backward",
        label="Copy backward",
        description="Tensor.copy_ backward",
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


def _fused_attention_score_backward_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        query = torch.randn(
            (params["batch"], params["heads"], params["sequence"], params["head_dim"]),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        key = torch.randn(
            (params["batch"], params["heads"], params["sequence"], params["head_dim"]),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        value = torch.randn(
            (params["batch"], params["heads"], params["sequence"], params["head_dim"]),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        grad_output = torch.ones_like(query)

        def run() -> torch.Tensor | None:
            _clear_gradients(query, key, value)
            result = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            result.backward(grad_output)
            _synchronize(device)
            return query.grad

        return run

    return BenchmarkSpec(
        name="fused_attention_score_backward",
        label="Fused attention backward",
        description="torch.nn.functional.scaled_dot_product_attention backward",
        cases=_cases_for("fused_attention_score"),
        build=build,
    )


def _triton_flash_attention_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        query = torch.randn(
            (params["sequence"], params["heads"], params["head_dim"]),
            device=device,
            dtype=dtype,
        )
        key = torch.randn(
            (params["sequence"], params["heads"], params["head_dim"]),
            device=device,
            dtype=dtype,
        )
        value = torch.randn(
            (params["sequence"], params["heads"], params["head_dim"]),
            device=device,
            dtype=dtype,
        )

        def run() -> torch.Tensor:
            result = triton_flash_attention(query, key, value, is_causal=False)
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="triton_flash_attention",
        label="Triton Flash Attention",
        description="custom Triton Flash Attention forward kernel",
        cases=_cases_for("triton_flash_attention"),
        build=build,
    )


def _triton_swiglu_spec() -> BenchmarkSpec:
    def build(device: torch.device, dtype: torch.dtype, params: dict[str, Any]) -> BenchmarkRunner:
        x = torch.randn((params["tokens"], params["hidden"]), device=device, dtype=dtype)
        w1 = torch.randn((params["intermediate"], params["hidden"]), device=device, dtype=dtype)
        w2 = torch.randn((params["intermediate"], params["hidden"]), device=device, dtype=dtype)
        w3 = torch.randn((params["hidden"], params["intermediate"]), device=device, dtype=dtype)

        def run() -> torch.Tensor:
            result = triton_swiglu(x, w1, w2, w3)
            _synchronize(device)
            return result

        return run

    return BenchmarkSpec(
        name="triton_swiglu",
        label="Triton SwiGLU",
        description="custom Triton fused SwiGLU forward kernel",
        cases=_cases_for("triton_swiglu"),
        build=build,
    )


BENCHMARK_SPECS = {
    spec.name: spec
    for spec in (
        _addmm_spec(),
        _addmm_backward_spec(),
        _bmm_spec(),
        _bmm_backward_spec(),
        _group_gemm_spec(),
        _group_gemm_backward_spec(),
        _layernorm_spec(),
        _layernorm_backward_spec(),
        _sum_spec(),
        _sum_backward_spec(),
        _concat_spec(),
        _concat_backward_spec(),
        _copy_spec(),
        _copy_backward_spec(),
        _fused_attention_score_spec(),
        _fused_attention_score_backward_spec(),
        _triton_flash_attention_spec(),
        _triton_swiglu_spec(),
    )
}


def list_benchmarks() -> list[BenchmarkSpec]:
    return [BENCHMARK_SPECS[name] for name in sorted(BENCHMARK_SPECS)]
