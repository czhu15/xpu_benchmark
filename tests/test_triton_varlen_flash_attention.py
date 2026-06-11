from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from xpu_benchmark.triton_varlen_flash_attention import (
    is_triton_varlen_flash_attention_available,
    triton_varlen_flash_attention,
)


def _available_accelerator() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return None


def _single_sequence_metadata(sequence: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cu_seqlens = torch.tensor([0, sequence], device=device, dtype=torch.int32)
    attn_arg = torch.zeros((sequence,), device=device, dtype=torch.int32)
    return attn_arg, attn_arg, cu_seqlens, cu_seqlens


class TritonVarlenFlashAttentionTests(unittest.TestCase):
    def test_matches_scaled_dot_product_attention_when_available(self) -> None:
        device = _available_accelerator()
        if not is_triton_varlen_flash_attention_available() or device is None:
            self.skipTest("Triton Flash Attention requires Triton and a CUDA/XPU device.")

        torch.manual_seed(0)
        sequence = 128
        query = torch.randn((sequence, 3, 32), device=device, dtype=torch.float16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k = _single_sequence_metadata(sequence, device)

        actual = triton_varlen_flash_attention(
            query,
            key,
            value,
            q_attn_arg,
            k_attn_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            sequence,
            sequence,
            mask_fn=0,
        )
        expected = F.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(0, 1)

        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=2.0e-2, rtol=2.0e-2))

    def test_matches_causal_scaled_dot_product_attention_when_available(self) -> None:
        device = _available_accelerator()
        if not is_triton_varlen_flash_attention_available() or device is None:
            self.skipTest("Triton Flash Attention requires Triton and a CUDA/XPU device.")

        torch.manual_seed(1)
        sequence = 128
        query = torch.randn((sequence, 2, 32), device=device, dtype=torch.float16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k = _single_sequence_metadata(sequence, device)

        actual = triton_varlen_flash_attention(
            query,
            key,
            value,
            q_attn_arg,
            k_attn_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            sequence,
            sequence,
            mask_fn=2,
        )
        expected = F.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            dropout_p=0.0,
            is_causal=True,
        ).transpose(0, 1)

        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=2.0e-2, rtol=2.0e-2))


if __name__ == "__main__":
    unittest.main()