from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from xpu_benchmark.triton_swiglu import is_triton_swiglu_available, triton_swiglu


def _available_accelerator() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return None


class TritonSwiGLUTests(unittest.TestCase):
    def test_matches_reference_swiglu_when_available(self) -> None:
        device = _available_accelerator()
        if not is_triton_swiglu_available() or device is None:
            self.skipTest("Triton SwiGLU requires Triton and a CUDA/XPU device.")

        torch.manual_seed(0)
        x = torch.randn((16, 32), device=device, dtype=torch.float16)
        w1 = torch.randn((64, 32), device=device, dtype=torch.float16)
        w2 = torch.randn((64, 32), device=device, dtype=torch.float16)
        w3 = torch.randn((32, 64), device=device, dtype=torch.float16)
        bias1 = torch.randn((64,), device=device, dtype=torch.float16)
        bias2 = torch.randn((64,), device=device, dtype=torch.float16)
        bias3 = torch.randn((32,), device=device, dtype=torch.float16)

        actual = triton_swiglu(x, w1, w2, w3, bias1, bias2, bias3)
        expected = F.linear(F.silu(F.linear(x, w1, bias1)) * F.linear(x, w2, bias2), w3, bias3)

        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=8.0e-2, rtol=8.0e-2))

    def test_matches_reference_swiglu_without_bias_when_available(self) -> None:
        device = _available_accelerator()
        if not is_triton_swiglu_available() or device is None:
            self.skipTest("Triton SwiGLU requires Triton and a CUDA/XPU device.")

        torch.manual_seed(1)
        x = torch.randn((12, 2, 24), device=device, dtype=torch.float16)[:, 0, :]
        w1 = torch.randn((48, 24), device=device, dtype=torch.float16)
        w2 = torch.randn((48, 24), device=device, dtype=torch.float16)
        w3 = torch.randn((24, 48), device=device, dtype=torch.float16)

        actual = triton_swiglu(x, w1, w2, w3)
        expected = F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w2), w3)

        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=8.0e-2, rtol=8.0e-2))


if __name__ == "__main__":
    unittest.main()