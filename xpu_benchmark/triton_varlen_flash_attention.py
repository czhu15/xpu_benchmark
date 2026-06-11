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


def is_triton_varlen_flash_attention_available() -> bool:
    return triton is not None and tl is not None and hasattr(tl, "make_tensor_descriptor")


def _is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


if is_triton_varlen_flash_attention_available():

    def _forward_configs() -> list[triton.Config]:
        if _is_xpu():
            return [
                triton.Config(
                    {"BLOCK_M": 128, "BLOCK_N": 32, "grf_mode": "256"},
                    num_stages=2,
                    num_warps=16,
                )
            ]
        return [triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=2, num_warps=4)]


    @triton.jit
    def _attention_mask(query_group, key_group, query_offsets, key_offsets, MASK_KIND: tl.constexpr):
        lower_triangle = query_offsets[:, None] >= key_offsets[None, :]
        upper_triangle = query_offsets[:, None] <= key_offsets[None, :]
        same_group = (query_group[:, None] == key_group[None, :]) | (key_group[None, :] == 0)
        diagonal = query_offsets[:, None] == key_offsets[None, :]
        if MASK_KIND == 1:
            return (upper_triangle & same_group) | diagonal
        if MASK_KIND == 2:
            return (lower_triangle & same_group) | diagonal
        return upper_triangle


    @triton.autotune(
        configs=_forward_configs(),
        key=["QK_DIM", "VALUE_DIM", "MASK_KIND", "SPARSE_OPT"],
    )
    @triton.jit
    def _varlen_flash_attention_forward_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        query_group_ptr,
        key_group_ptr,
        cu_query_lengths,
        cu_key_lengths,
        query_heads: tl.constexpr,
        key_heads: tl.constexpr,
        softmax_scale,
        QK_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        MASK_KIND: tl.constexpr,
        SPARSE_OPT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_tile = tl.program_id(0)
        query_head = tl.program_id(1)
        batch = tl.program_id(2)
        key_head = query_head // (query_heads // key_heads)

        query_start = tl.load(cu_query_lengths + batch)
        query_end = tl.load(cu_query_lengths + batch + 1)
        query_length = query_end - query_start
        row_start = query_tile * BLOCK_M
        if row_start >= query_length:
            return

        key_start = tl.load(cu_key_lengths + batch)
        key_end = tl.load(cu_key_lengths + batch + 1)
        key_length = key_end - key_start
        if key_length == 0:
            return

        if SPARSE_OPT:
            col_begin = 0
            col_end = key_length
        elif MASK_KIND == 0:
            col_begin = 0
            col_end = key_length
        elif MASK_KIND & 1:
            col_begin = row_start
            if col_begin >= key_length:
                return
            col_end = key_length
        else:
            col_begin = 0
            col_end = tl.minimum(row_start + BLOCK_M, key_length)

        log2e: tl.constexpr = 1.4426950408889634
        scaled_log2e = softmax_scale.to(tl.float32) * log2e
        row_offsets = row_start + tl.arange(0, BLOCK_M)

        query_start = query_start.to(tl.int64)
        key_start = key_start.to(tl.int64)
        query_base = query_ptr + query_start * query_heads * QK_DIM + query_head * QK_DIM
        key_base = key_ptr + key_start * key_heads * QK_DIM + key_head * QK_DIM
        value_base = value_ptr + key_start * key_heads * VALUE_DIM + key_head * VALUE_DIM
        output_base = output_ptr + query_start * query_heads * VALUE_DIM + query_head * VALUE_DIM

        query_desc = tl.make_tensor_descriptor(
            query_base,
            shape=[query_length, QK_DIM],
            strides=[query_heads * QK_DIM, 1],
            block_shape=[BLOCK_M, QK_DIM],
        )
        key_desc = tl.make_tensor_descriptor(
            key_base,
            shape=[key_length, QK_DIM],
            strides=[key_heads * QK_DIM, 1],
            block_shape=[BLOCK_N, QK_DIM],
        )
        value_desc = tl.make_tensor_descriptor(
            value_base,
            shape=[key_length, VALUE_DIM],
            strides=[key_heads * VALUE_DIM, 1],
            block_shape=[BLOCK_N, VALUE_DIM],
        )
        output_desc = tl.make_tensor_descriptor(
            output_base,
            shape=[query_length, VALUE_DIM],
            strides=[query_heads * VALUE_DIM, 1],
            block_shape=[BLOCK_M, VALUE_DIM],
        )

        if MASK_KIND != 0 and MASK_KIND != 3:
            query_group_desc = tl.make_tensor_descriptor(
                query_group_ptr + query_start,
                shape=[query_length],
                strides=[1],
                block_shape=[BLOCK_M],
            )
            key_group_desc = tl.make_tensor_descriptor(
                key_group_ptr + key_start,
                shape=[key_length],
                strides=[1],
                block_shape=[BLOCK_N],
            )
            query_group = query_group_desc.load([row_start])

        query_block = query_desc.load([row_start, 0])
        output_accumulator = tl.zeros((BLOCK_M, VALUE_DIM), dtype=tl.float32)
        row_max = tl.full((BLOCK_M,), -3.4028234663852886e38, dtype=tl.float32)
        row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for col_start in tl.range(col_begin, col_end, BLOCK_N):
            col_start = tl.multiple_of(col_start, BLOCK_N).to(tl.int32)
            col_offsets = col_start + tl.arange(0, BLOCK_N)

            if MASK_KIND == 0:
                score_mask = col_offsets[None, :] < key_length
            elif MASK_KIND == 3:
                score_mask = row_offsets[:, None] <= col_offsets[None, :]
            else:
                key_group = key_group_desc.load([col_start])
                score_mask = _attention_mask(query_group, key_group, row_offsets, col_offsets, MASK_KIND)

            key_block = key_desc.load([col_start, 0]).T
            scores = tl.dot(query_block, key_block)
            scores = tl.where(score_mask & (col_offsets[None, :] < key_length), scores, -3.4028234663852886e38)

            next_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
            previous_scale = tl.math.exp2((row_max - next_row_max) * scaled_log2e)
            probabilities = tl.math.exp2((scores - next_row_max[:, None]) * scaled_log2e)
            output_accumulator *= previous_scale[:, None]

            value_block = value_desc.load([col_start, 0])
            output_accumulator += tl.dot(probabilities.to(value_block.dtype), value_block)
            row_sum = row_sum * previous_scale + tl.sum(probabilities, axis=1)
            row_max = next_row_max

        output = output_accumulator / row_sum[:, None]
        output_desc.store([row_start, 0], output.to(output_ptr.type.element_ty))


def _validate_flash_attention_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_attn_arg: torch.Tensor,
    k_attn_arg: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> None:
    if not is_triton_varlen_flash_attention_available():
        raise RuntimeError("Triton with tensor descriptors is required for triton_varlen_flash_attention, but it is not installed.")
    if query.device.type not in {"cuda", "xpu"}:
        raise RuntimeError("triton_varlen_flash_attention requires a CUDA or XPU tensor device.")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must be 3D tensors: (total_tokens, heads, head_dim).")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match.")
    if key.shape[:2] != value.shape[:2]:
        raise ValueError("key and value must have matching total tokens and heads.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by key/value head count.")
    if query.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("triton_varlen_flash_attention supports float16, bfloat16, and float32 tensors.")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("query, key, and value must use the same dtype.")
    for tensor in (key, value, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k):
        if tensor.device != query.device:
            raise ValueError("all FlashAttention inputs must be on the same device.")
    if q_attn_arg.ndim != 1 or q_attn_arg.shape[0] != query.shape[0]:
        raise ValueError("q_attn_arg must have shape (total_q,).")
    if k_attn_arg.ndim != 1 or k_attn_arg.shape[0] != key.shape[0]:
        raise ValueError("k_attn_arg must have shape (total_k,).")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1 or cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1D tensors with matching shape.")
    if int(cu_seqlens_q[-1].item()) != query.shape[0] or int(cu_seqlens_k[-1].item()) != key.shape[0]:
        raise ValueError("cumulative sequence lengths must end at total query/key token counts.")
    if query.shape[-1] > 256 or value.shape[-1] > 256:
        raise ValueError("triton_varlen_flash_attention currently supports qk/value head dimensions <= 256.")


def triton_varlen_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_attn_arg: torch.Tensor,
    k_attn_arg: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    mask_fn: int = 1,
    sparse_opt: bool = False,
) -> torch.Tensor:
    """Run forward-only varlen FlashAttention with reference-compatible metadata inputs."""
    _validate_flash_attention_inputs(query, key, value, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k)

    total_q, query_heads, qk_dim = query.shape
    _, key_heads, value_dim = value.shape
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(qk_dim)
    query_contiguous = query.contiguous()
    key_contiguous = key.contiguous()
    value_contiguous = value.contiguous()
    output = torch.empty((total_q, query_heads, value_dim), device=query.device, dtype=query.dtype)
    batch = cu_seqlens_q.shape[0] - 1

    grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), query_heads, batch)
    _varlen_flash_attention_forward_kernel[grid](
        query_contiguous,
        key_contiguous,
        value_contiguous,
        output,
        q_attn_arg,
        k_attn_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        query_heads,
        key_heads,
        scale,
        QK_DIM=qk_dim,
        VALUE_DIM=value_dim,
        MASK_KIND=mask_fn,
        SPARSE_OPT=sparse_opt,
    )
    return output


__all__ = ["is_triton_varlen_flash_attention_available", "triton_varlen_flash_attention"]
