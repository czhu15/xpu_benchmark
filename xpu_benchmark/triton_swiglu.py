from __future__ import annotations

# pyright: reportInvalidTypeForm=false, reportMissingImports=false

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised when Triton is unavailable.
    triton = None
    tl = None


def is_triton_swiglu_available() -> bool:
    return triton is not None and tl is not None


if is_triton_swiglu_available():

    @triton.jit
    def _swiglu_forward_kernel(
        x,
        w1,
        w2,
        w3,
        bias1,
        bias2,
        bias3,
        output,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        intermediate: tl.constexpr,
        stride_xt: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_w1i: tl.constexpr,
        stride_w1h: tl.constexpr,
        stride_w2i: tl.constexpr,
        stride_w2h: tl.constexpr,
        stride_w3h: tl.constexpr,
        stride_w3i: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_oh: tl.constexpr,
        has_bias1: tl.constexpr,
        has_bias2: tl.constexpr,
        has_bias3: tl.constexpr,
        block_tokens: tl.constexpr,
        block_hidden_out: tl.constexpr,
        block_intermediate: tl.constexpr,
        block_hidden_in: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * block_tokens + tl.arange(0, block_tokens)
        hidden_out_offsets = tl.program_id(1) * block_hidden_out + tl.arange(0, block_hidden_out)
        intermediate_offsets_base = tl.arange(0, block_intermediate)
        hidden_in_offsets_base = tl.arange(0, block_hidden_in)

        output_accumulator = tl.zeros((block_tokens, block_hidden_out), tl.float32)

        for intermediate_start in range(0, intermediate, block_intermediate):
            intermediate_offsets = intermediate_start + intermediate_offsets_base
            gate_accumulator = tl.zeros((block_tokens, block_intermediate), tl.float32)
            up_accumulator = tl.zeros((block_tokens, block_intermediate), tl.float32)

            for hidden_start in range(0, hidden, block_hidden_in):
                hidden_in_offsets = hidden_start + hidden_in_offsets_base
                x_block = tl.load(
                    x + token_offsets[:, None] * stride_xt + hidden_in_offsets[None, :] * stride_xh,
                    mask=(token_offsets[:, None] < tokens) & (hidden_in_offsets[None, :] < hidden),
                    other=0.0,
                )
                w1_block = tl.load(
                    w1 + intermediate_offsets[None, :] * stride_w1i + hidden_in_offsets[:, None] * stride_w1h,
                    mask=(intermediate_offsets[None, :] < intermediate) & (hidden_in_offsets[:, None] < hidden),
                    other=0.0,
                )
                w2_block = tl.load(
                    w2 + intermediate_offsets[None, :] * stride_w2i + hidden_in_offsets[:, None] * stride_w2h,
                    mask=(intermediate_offsets[None, :] < intermediate) & (hidden_in_offsets[:, None] < hidden),
                    other=0.0,
                )
                gate_accumulator += tl.dot(x_block, w1_block)
                up_accumulator += tl.dot(x_block, w2_block)

            if has_bias1:
                gate_accumulator += tl.load(
                    bias1 + intermediate_offsets,
                    mask=intermediate_offsets < intermediate,
                    other=0.0,
                )[None, :]
            if has_bias2:
                up_accumulator += tl.load(
                    bias2 + intermediate_offsets,
                    mask=intermediate_offsets < intermediate,
                    other=0.0,
                )[None, :]

            silu_values = gate_accumulator / (1.0 + tl.exp(-gate_accumulator))
            fused_values = silu_values * up_accumulator
            w3_block = tl.load(
                w3 + hidden_out_offsets[None, :] * stride_w3h + intermediate_offsets[:, None] * stride_w3i,
                mask=(hidden_out_offsets[None, :] < hidden) & (intermediate_offsets[:, None] < intermediate),
                other=0.0,
            )
            output_accumulator += tl.dot(fused_values.to(w3_block.dtype), w3_block)

        if has_bias3:
            output_accumulator += tl.load(
                bias3 + hidden_out_offsets,
                mask=hidden_out_offsets < hidden,
                other=0.0,
            )[None, :]

        tl.store(
            output + token_offsets[:, None] * stride_ot + hidden_out_offsets[None, :] * stride_oh,
            output_accumulator,
            mask=(token_offsets[:, None] < tokens) & (hidden_out_offsets[None, :] < hidden),
        )


def _validate_swiglu_inputs(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    bias3: torch.Tensor | None,
) -> None:
    if not is_triton_swiglu_available():
        raise RuntimeError("Triton is required for triton_swiglu, but it is not installed.")
    if x.device.type not in {"cuda", "xpu"}:
        raise RuntimeError("triton_swiglu requires a CUDA or XPU tensor device.")
    if x.ndim < 2:
        raise ValueError("x must have shape (..., hidden).")
    if w1.ndim != 2 or w2.ndim != 2 or w3.ndim != 2:
        raise ValueError("w1, w2, and w3 must be 2D linear weights.")

    hidden = x.shape[-1]
    intermediate = w1.shape[0]
    if w1.shape != w2.shape:
        raise ValueError("w1 and w2 must have identical shape: (intermediate, hidden).")
    if w1.shape[1] != hidden:
        raise ValueError("w1 and w2 input dimension must match x.shape[-1].")
    if w3.shape != (hidden, intermediate):
        raise ValueError("w3 must have shape (hidden, intermediate).")

    tensors = [x, w1, w2, w3]
    optional_tensors = [bias for bias in (bias1, bias2, bias3) if bias is not None]
    for tensor in tensors + optional_tensors:
        if tensor.device != x.device:
            raise ValueError("x, weights, and biases must be on the same device.")
        if tensor.dtype != x.dtype:
            raise TypeError("x, weights, and biases must use the same dtype.")
    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("triton_swiglu supports float16, bfloat16, and float32 tensors.")
    if bias1 is not None and bias1.shape != (intermediate,):
        raise ValueError("bias1 must have shape (intermediate,).")
    if bias2 is not None and bias2.shape != (intermediate,):
        raise ValueError("bias2 must have shape (intermediate,).")
    if bias3 is not None and bias3.shape != (hidden,):
        raise ValueError("bias3 must have shape (hidden,).")


def triton_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    bias3: torch.Tensor | None = None,
    *,
    block_tokens: int = 16,
    block_hidden_out: int = 32,
    block_intermediate: int = 32,
    block_hidden_in: int = 32,
) -> torch.Tensor:
    """Run full SwiGLU forward with one Triton fused kernel.

    Computes ``linear(silu(linear(x, w1, bias1)) * linear(x, w2, bias2), w3, bias3)``.
    """
    _validate_swiglu_inputs(x, w1, w2, w3, bias1, bias2, bias3)

    hidden = x.shape[-1]
    intermediate = w1.shape[0]
    x_flat = x.reshape(-1, hidden).contiguous()
    w1_contiguous = w1.contiguous()
    w2_contiguous = w2.contiguous()
    w3_contiguous = w3.contiguous()
    bias1_contiguous = bias1.contiguous() if bias1 is not None else x_flat
    bias2_contiguous = bias2.contiguous() if bias2 is not None else x_flat
    bias3_contiguous = bias3.contiguous() if bias3 is not None else x_flat
    output = torch.empty_like(x_flat)

    grid = (triton.cdiv(x_flat.shape[0], block_tokens), triton.cdiv(hidden, block_hidden_out))
    _swiglu_forward_kernel[grid](
        x_flat,
        w1_contiguous,
        w2_contiguous,
        w3_contiguous,
        bias1_contiguous,
        bias2_contiguous,
        bias3_contiguous,
        output,
        x_flat.shape[0],
        hidden,
        intermediate,
        x_flat.stride(0),
        x_flat.stride(1),
        w1_contiguous.stride(0),
        w1_contiguous.stride(1),
        w2_contiguous.stride(0),
        w2_contiguous.stride(1),
        w3_contiguous.stride(0),
        w3_contiguous.stride(1),
        output.stride(0),
        output.stride(1),
        bias1 is not None,
        bias2 is not None,
        bias3 is not None,
        block_tokens,
        block_hidden_out,
        block_intermediate,
        block_hidden_in,
        num_warps=4,
    )
    return output.reshape(x.shape)


__all__ = ["is_triton_swiglu_available", "triton_swiglu"]