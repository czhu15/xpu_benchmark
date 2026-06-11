from __future__ import annotations

# pyright: reportInvalidTypeForm=false, reportMissingImports=false

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - exercised when Triton is unavailable.
    triton = None
    tl = None
    libdevice = None


def is_triton_swiglu_available() -> bool:
    return triton is not None and tl is not None and libdevice is not None and hasattr(tl, "make_tensor_descriptor")


def _is_ampere() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 8


def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 9


def _is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


if is_triton_swiglu_available():

    @triton.jit
    def _fast_silu(x):
        dtype = x.type.element_ty
        x = x.to(tl.float32)
        return libdevice.fast_dividef(x, 1.0 + libdevice.fast_expf(-x)).to(dtype)


    def _get_autotune_configs() -> list[triton.Config]:
        if _is_hopper():
            return [
                triton.Config(
                    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                    num_stages=3,
                    num_warps=4,
                )
            ]
        if _is_ampere():
            return [
                triton.Config(
                    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
                    num_stages=3,
                    num_warps=4,
                )
            ]
        if _is_xpu():
            return [
                triton.Config(
                    {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                    num_stages=3,
                    num_warps=16,
                )
            ]
        return [
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=3,
                num_warps=4,
            )
        ]


    @triton.autotune(
        configs=_get_autotune_configs(),
        key=["N", "K", "IS_TRAINING"],
    )
    @triton.jit
    def _fused_swiglu_fwd_kernel(
        x_ptr,
        w_g_ptr,
        w_fc_ptr,
        b_g_ptr,
        b_fc_ptr,
        y_ptr,
        g_ptr,
        fc_ptr,
        M,
        N,
        K,
        IS_TRAINING: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        dtype = y_ptr.type.element_ty
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N):
            return

        desc_x = tl.make_tensor_descriptor(
            x_ptr,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        desc_wg = tl.make_tensor_descriptor(
            w_g_ptr,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        )
        desc_wfc = tl.make_tensor_descriptor(
            w_fc_ptr,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        )

        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N
        offset_n = off_n + tl.arange(0, BLOCK_SIZE_N)
        b_g = tl.load(b_g_ptr + offset_n, mask=offset_n < N, other=0.0)
        b_fc = tl.load(b_fc_ptr + offset_n, mask=offset_n < N, other=0.0)

        accumulator_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        accumulator_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        off_k = 0
        for k in tl.range(0, K, BLOCK_SIZE_K):
            k = tl.multiple_of(k, BLOCK_SIZE_K)
            x = desc_x.load([off_m, off_k])
            w_g = desc_wg.load([k, off_n])
            w_fc = desc_wfc.load([k, off_n])
            accumulator_g = tl.dot(x, w_g, accumulator_g)
            accumulator_fc = tl.dot(x, w_fc, accumulator_fc)
            off_k += BLOCK_SIZE_K

        accumulator_g += b_g[None, :]
        accumulator_fc += b_fc[None, :]
        accumulator_g = accumulator_g.to(dtype)
        accumulator_fc = accumulator_fc.to(dtype)
        silu_g = _fast_silu(accumulator_g)
        y = (silu_g.to(tl.float32) * accumulator_fc.to(tl.float32)).to(dtype)

        desc_y = tl.make_tensor_descriptor(
            y_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )
        desc_y.store([off_m, off_n], y)

        if IS_TRAINING:
            desc_g = tl.make_tensor_descriptor(
                g_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )
            desc_fc = tl.make_tensor_descriptor(
                fc_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )
            desc_g.store([off_m, off_n], accumulator_g)
            desc_fc.store([off_m, off_n], accumulator_fc)


def _validate_swiglu_inputs(
    x: torch.Tensor,
    w_g: torch.Tensor,
    w_fc: torch.Tensor,
    b_g: torch.Tensor | None,
    b_fc: torch.Tensor | None,
) -> None:
    if not is_triton_swiglu_available():
        raise RuntimeError("Triton with tensor descriptors is required for triton_swiglu, but it is not installed.")
    if x.device.type not in {"cuda", "xpu"}:
        raise RuntimeError("triton_swiglu requires a CUDA or XPU tensor device.")
    if x.ndim < 2:
        raise ValueError("x must have shape (..., hidden).")
    if w_g.ndim != 2 or w_fc.ndim != 2:
        raise ValueError("w_g and w_fc must be 2D weights with shape (hidden, intermediate).")

    hidden = x.shape[-1]
    intermediate = w_g.shape[1]
    if w_g.shape != w_fc.shape:
        raise ValueError("w_g and w_fc must have identical shape: (hidden, intermediate).")
    if w_g.shape[0] != hidden:
        raise ValueError("w_g and w_fc input dimension must match x.shape[-1].")

    tensors = [x, w_g, w_fc]
    optional_tensors = [bias for bias in (b_g, b_fc) if bias is not None]
    for tensor in tensors + optional_tensors:
        if tensor.device != x.device:
            raise ValueError("x, weights, and biases must be on the same device.")
        if tensor.dtype != x.dtype:
            raise TypeError("x, weights, and biases must use the same dtype.")
    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("triton_swiglu supports float16, bfloat16, and float32 tensors.")
    if b_g is not None and b_g.shape != (intermediate,):
        raise ValueError("b_g must have shape (intermediate,).")
    if b_fc is not None and b_fc.shape != (intermediate,):
        raise ValueError("b_fc must have shape (intermediate,).")


def triton_swiglu(
    x: torch.Tensor,
    w_g: torch.Tensor,
    w_fc: torch.Tensor,
    b_g: torch.Tensor | None = None,
    b_fc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the optimized fused two-GEMM SwiGLU forward kernel.

    Computes ``silu(x @ w_g + b_g) * (x @ w_fc + b_fc)`` where weights use
    matrix-multiply layout ``(hidden, intermediate)``.
    """
    _validate_swiglu_inputs(x, w_g, w_fc, b_g, b_fc)

    hidden = x.shape[-1]
    intermediate = w_g.shape[1]
    x_flat = x.reshape(-1, hidden).contiguous()
    tokens = x_flat.shape[0]

    w_g_contiguous = w_g.contiguous()
    w_fc_contiguous = w_fc.contiguous()
    b_g_contiguous = b_g.contiguous() if b_g is not None else torch.zeros((intermediate,), device=x.device, dtype=x.dtype)
    b_fc_contiguous = b_fc.contiguous() if b_fc is not None else torch.zeros((intermediate,), device=x.device, dtype=x.dtype)
    output = torch.empty((tokens, intermediate), device=x.device, dtype=x.dtype)
    gate = x.new_empty(1)
    fc = x.new_empty(1)

    total_len = tokens
    if intermediate % 64 != 0 or hidden % 32 != 0:
        raise ValueError(
            "triton_swiglu tensor descriptor kernel requires intermediate to be divisible by 64 "
            "and hidden to be divisible by 32."
        )

    grid = lambda META: (
        triton.cdiv(total_len, META["BLOCK_SIZE_M"]) * triton.cdiv(intermediate, META["BLOCK_SIZE_N"]),
    )
    _fused_swiglu_fwd_kernel[grid](
        x_flat,
        w_g_contiguous,
        w_fc_contiguous,
        b_g_contiguous,
        b_fc_contiguous,
        output,
        gate,
        fc,
        total_len,
        intermediate,
        hidden,
        IS_TRAINING=False,
    )
    return output.reshape(*x.shape[:-1], intermediate)


__all__ = ["is_triton_swiglu_available", "triton_swiglu"]
