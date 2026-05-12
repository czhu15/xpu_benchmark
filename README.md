# xpu_benchmark

`xpu_benchmark` is a small benchmark repo for running representative PyTorch ops on Intel XPU with either `torch.utils.benchmark.Timer` or Python's `timeit.Timer`.

## Covered ops

- `addmm`
- `bmm`
- `group_gemm`
- `layernorm`
- `sum`
- `concat`
- `copy`
- `fused_attention_score`

## Notes on op mappings

- `group_gemm` is implemented as a grouped workload of independent `torch.matmul` calls, measured as one benchmark case.
- `fused_attention_score` is benchmarked through `torch.nn.functional.scaled_dot_product_attention`, which is the closest fused attention primitive exposed directly through `torch`.

## Requirements

- Python with the `torch` package already installed
- Intel XPU available through `torch.xpu`

## Usage

List the available benchmarks:

```bash
python -m xpu_benchmark list
```

Run all benchmarks on XPU:

```bash
python -m xpu_benchmark run --device xpu
```

Run with Python's built-in `timeit.Timer` instead of the default PyTorch timer:

```bash
python -m xpu_benchmark run --device xpu --timer timeit
```

Run a subset and save JSON output:

```bash
python -m xpu_benchmark run --ops addmm bmm fused_attention_score --device xpu --dtype float16 --timer torch --format json --output results/run.json
```

## Default measurement flow

The default harness follows the PyTorch benchmark recipe pattern:

1. Create benchmark inputs on the selected device.
2. Wrap the benchmark callable with `torch.utils.benchmark.Timer`.
3. Measure with `blocked_autorange`.
4. Emit either a comparison table or JSON summary.

When `--timer timeit` is selected, the harness wraps the same benchmark callable with `timeit.Timer`, chooses an iteration count using an autorange-style loop, then reports median, mean, and IQR from repeated per-run timings. The benchmark callables synchronize XPU/CUDA work before returning, so accelerator timings include kernel completion rather than only launch overhead.

The default shapes are intentionally moderate so the suite is easy to extend without rewriting the harness.
