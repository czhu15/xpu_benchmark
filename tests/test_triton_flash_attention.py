from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from xpu_benchmark.triton_flash_attention import is_triton_flash_attention_available, triton_flash_attention


def _available_accelerator() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return None


class TritonFlashAttentionTests(unittest.TestCase):
    def test_matches_scaled_dot_product_attention_when_available(self) -> None:
        device = _available_accelerator()
        if not is_triton_flash_attention_available() or device is None:
            self.skipTest("Triton Flash Attention requires Triton and a CUDA/XPU device.")

        torch.manual_seed(0)
        query = torch.randn((32, 3, 32), device=device, dtype=torch.float16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        actual = triton_flash_attention(query, key, value)
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
        if not is_triton_flash_attention_available() or device is None:
            self.skipTest("Triton Flash Attention requires Triton and a CUDA/XPU device.")

        torch.manual_seed(1)
        query = torch.randn((32, 2, 32), device=device, dtype=torch.float16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        actual = triton_flash_attention(query, key, value, is_causal=True)
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