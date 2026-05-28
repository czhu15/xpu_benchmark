from __future__ import annotations

# pyright: reportInvalidTypeForm=false, reportMissingImports=false

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised when Triton is unavailable.
    triton = None
    tl = None


def is_triton_flash_attention_available() -> bool:
    return triton is not None and tl is not None


if is_triton_flash_attention_available():

    @triton.jit
    def _flash_attention_forward_kernel(
        query,
        key,
        value,
        output,
        softmax_scale: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_od: tl.constexpr,
        heads: tl.constexpr,
        sequence: tl.constexpr,
        head_dim: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        block_start_m = tl.program_id(0) * block_m
        batch_head = tl.program_id(1)
        head = batch_head % heads
        batch = batch_head // heads

        offs_m = block_start_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)

        query_ptrs = (
            query
            + batch * stride_qb
            + head * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
        query_block = tl.load(
            query_ptrs,
            mask=(offs_m[:, None] < sequence) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        max_score = tl.full((block_m,), -float("inf"), tl.float32)
        normalizer = tl.zeros((block_m,), tl.float32)
        accumulator = tl.zeros((block_m, block_d), tl.float32)

        for start_n in range(0, sequence, block_n):
            current_n = start_n + offs_n
            key_ptrs = (
                key
                + batch * stride_kb
                + head * stride_kh
                + current_n[None, :] * stride_kn
                + offs_d[:, None] * stride_kd
            )
            value_ptrs = (
                value
                + batch * stride_vb
                + head * stride_vh
                + current_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            key_block = tl.load(
                key_ptrs,
                mask=(current_n[None, :] < sequence) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            value_block = tl.load(
                value_ptrs,
                mask=(current_n[:, None] < sequence) & (offs_d[None, :] < head_dim),
                other=0.0,
            )

            scores = tl.dot(query_block, key_block) * softmax_scale
            valid_mask = current_n[None, :] < sequence
            if is_causal:
                valid_mask = valid_mask & (offs_m[:, None] >= current_n[None, :])
            scores = tl.where(valid_mask, scores, -float("inf"))

            new_max_score = tl.maximum(max_score, tl.max(scores, axis=1))
            probabilities = tl.exp(scores - new_max_score[:, None])
            correction = tl.exp(max_score - new_max_score)
            new_normalizer = normalizer * correction + tl.sum(probabilities, axis=1)

            probabilities = probabilities.to(value_block.dtype)
            accumulator = accumulator * correction[:, None] + tl.dot(probabilities, value_block)
            max_score = new_max_score
            normalizer = new_normalizer

        output_block = accumulator / normalizer[:, None]
        output_ptrs = (
            output
            + batch * stride_ob
            + head * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )
        tl.store(
            output_ptrs,
            output_block,
            mask=(offs_m[:, None] < sequence) & (offs_d[None, :] < head_dim),
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_flash_attention_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if not is_triton_flash_attention_available():
        raise RuntimeError("Triton is required for triton_flash_attention, but it is not installed.")
    if query.device.type not in {"cuda", "xpu"}:
        raise RuntimeError("triton_flash_attention requires a CUDA or XPU tensor device.")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shape: (batch, heads, sequence, head_dim).")
    if query.ndim != 4:
        raise ValueError("query, key, and value must be 4D tensors: (batch, heads, sequence, head_dim).")
    if query.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("triton_flash_attention supports float16, bfloat16, and float32 tensors.")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("query, key, and value must use the same dtype.")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key, and value must be on the same device.")
    if query.shape[-1] > 256:
        raise ValueError("triton_flash_attention currently supports head_dim <= 256.")


def triton_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool = False,
    softmax_scale: float | None = None,
    block_m: int = 16,
    block_n: int = 32,
) -> torch.Tensor:
    """Run Flash Attention forward with a Triton kernel.

    Inputs must have shape ``(batch, heads, sequence, head_dim)``. The function
    implements inference-style forward attention without dropout or backward.
    """
    _validate_flash_attention_inputs(query, key, value)

    batch, heads, sequence, head_dim = query.shape
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    block_d = _next_power_of_2(head_dim)
    output = torch.empty_like(query)

    grid = (triton.cdiv(sequence, block_m), batch * heads)
    _flash_attention_forward_kernel[grid](
        query,
        key,
        value,
        output,
        scale,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        heads,
        sequence,
        head_dim,
        block_m,
        block_n,
        block_d,
        is_causal,
        num_warps=4,
    )
    return output


__all__ = ["is_triton_flash_attention_available", "triton_flash_attention"]