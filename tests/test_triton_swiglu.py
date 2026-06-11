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
        x = torch.randn((257, 64), device=device, dtype=torch.float16)
        w_g = torch.randn((64, 128), device=device, dtype=torch.float16)
        w_fc = torch.randn((64, 128), device=device, dtype=torch.float16)
        bias1 = torch.randn((128,), device=device, dtype=torch.float16)
        bias2 = torch.randn((128,), device=device, dtype=torch.float16)

        actual = triton_swiglu(x, w_g, w_fc, bias1, bias2)
        expected = F.silu(x @ w_g + bias1) * (x @ w_fc + bias2)

        self.assertEqual(actual.shape, (257, 128))
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=8.0e-2, rtol=8.0e-2))

    def test_matches_reference_swiglu_without_bias_when_available(self) -> None:
        device = _available_accelerator()
        if not is_triton_swiglu_available() or device is None:
            self.skipTest("Triton SwiGLU requires Triton and a CUDA/XPU device.")

        torch.manual_seed(1)
        x = torch.randn((513, 2, 64), device=device, dtype=torch.float16)[:, 0, :]
        w_g = torch.randn((64, 128), device=device, dtype=torch.float16)
        w_fc = torch.randn((64, 128), device=device, dtype=torch.float16)

        actual = triton_swiglu(x, w_g, w_fc)
        expected = F.silu(x @ w_g) * (x @ w_fc)

        self.assertEqual(actual.shape, (513, 128))
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=8.0e-2, rtol=8.0e-2))


if __name__ == "__main__":
    unittest.main()