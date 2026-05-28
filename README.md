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
- `triton_flash_attention`

Each PyTorch op also has a backward benchmark named `<op>_backward`, for example `addmm_backward` and `fused_attention_score_backward`. `triton_flash_attention` is a custom forward-only Triton kernel.

## Notes on op mappings

- `group_gemm` is implemented as a grouped workload of independent `torch.matmul` calls, measured as one benchmark case.
- `fused_attention_score` is benchmarked through `torch.nn.functional.scaled_dot_product_attention`, which is the closest fused attention primitive exposed directly through `torch`.
- `triton_flash_attention` is implemented in `xpu_benchmark/triton_flash_attention.py` as a tiled online-softmax Flash Attention forward kernel for tensors shaped `(batch, heads, sequence, head_dim)`.

## Requirements

- Python with the `torch` package already installed
- Intel XPU available through `torch.xpu`
- Optional: `triton` for the `triton_flash_attention` benchmark

## Usage

List the available benchmarks:

```bash
python -m xpu_benchmark list
```

Run all benchmarks on XPU:

```bash
python -m xpu_benchmark run --device xpu
```

Run with the PyTorch benchmark timer instead of the default Python `timeit.Timer`:

```bash
python -m xpu_benchmark run --device xpu --timer torch
```

Run a subset and save JSON output:

```bash
python -m xpu_benchmark run --ops addmm bmm fused_attention_score --device xpu --dtype float16 --timer torch --format json --output results/run.json
```

Run with a fixed number of iterations per measurement instead of automatic autoranging:

```bash
python -m xpu_benchmark run --ops triton_flash_attention --device xpu --dtype float16 --runs 100
```

Run forward and backward benchmarks for the same op:

```bash
python -m xpu_benchmark run --ops addmm addmm_backward --device xpu
```

## Benchmark shape configuration

Benchmark input sizes are defined in `xpu_benchmark/benchmark_shapes.json`. Each top-level key is a forward op name, and its value is a list of parameter objects. Backward benchmarks reuse the corresponding forward op's cases. Add more objects to run the same op with multiple input shapes or parameter combinations.

For example, to benchmark two `addmm` shapes:

```json
{
	"addmm": [
		{"m": 512, "k": 512, "n": 512},
		{"m": 1024, "k": 512, "n": 256}
	]
}
```

When an op has multiple entries, `python -m xpu_benchmark run --ops <op>` runs all configured entries and prints one result row per entry.

## Measurement flow

By default, the harness wraps each benchmark callable with `timeit.Timer`, chooses an iteration count using an autorange-style loop, then reports median and mean from repeated per-run timings. The benchmark callables synchronize XPU/CUDA work before returning, so accelerator timings include kernel completion rather than only launch overhead.

When `--timer torch` is selected, the harness follows the PyTorch benchmark recipe pattern:

1. Create benchmark inputs on the selected device.
2. Wrap the benchmark callable with `torch.utils.benchmark.Timer`.
3. Measure with `blocked_autorange`.
4. Emit either a comparison table or JSON summary.

The default shapes in `benchmark_shapes.json` are intentionally moderate so the suite is easy to extend without rewriting the harness.
