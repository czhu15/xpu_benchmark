# xpu_benchmark

`xpu_benchmark` is a small benchmark repo for running representative PyTorch ops on Intel XPU with XPU/CUDA events, `torch.utils.benchmark.Timer`, or Python's `timeit.Timer`.

## Covered ops

- `addmm`
- `bmm`
- `group_gemm`
- `layernorm`
- `sum`
- `concat`
- `copy`
- `fused_attention_score`
- `triton_varlen_flash_attention`
- `triton_swiglu`

Each PyTorch op also has a backward benchmark named `<op>_backward`, for example `addmm_backward` and `fused_attention_score_backward`. `triton_varlen_flash_attention` and `triton_swiglu` are custom forward-only Triton kernels.

## Notes on op mappings

- `group_gemm` is implemented as a grouped workload of independent `torch.matmul` calls, measured as one benchmark case.
- `fused_attention_score` is benchmarked through `torch.nn.functional.scaled_dot_product_attention`, which is the closest fused attention primitive exposed directly through `torch`.
- `triton_varlen_flash_attention` is implemented in `xpu_benchmark/triton_varlen_flash_attention.py` as a varlen tiled online-softmax Flash Attention forward kernel. Its inputs follow the packed reference-style interface: `q`, `k`, `v`, `q_attn_arg`, `k_attn_arg`, `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`, `scale`, `mask_fn`, and `sparse_opt`.
- `triton_swiglu` is implemented in `xpu_benchmark/triton_swiglu.py` as the fused two-GEMM SwiGLU path from Intel Triton XPU PR 7152: `silu(x @ w_g + b_g) * (x @ w_fc + b_fc)`.

## Requirements

- Python with the `torch` package already installed
- Intel XPU available through `torch.xpu`
- Optional: `triton` for the `triton_varlen_flash_attention` benchmark

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

Measure accelerator-side kernel time with XPU/CUDA events:

```bash
python -m xpu_benchmark run --ops triton_swiglu --device xpu --dtype bfloat16 --timer event
```

Run a subset and save JSON output:

```bash
python -m xpu_benchmark run --ops addmm bmm fused_attention_score --device xpu --dtype float16 --timer torch --format json --output results/run.json
```

Run with a fixed number of iterations per measurement instead of automatic autoranging:

```bash
python -m xpu_benchmark run --ops triton_varlen_flash_attention --device xpu --dtype float16 --runs 100
```

Override the roofline reference peaks used for utilization percentages:

```bash
python -m xpu_benchmark run --ops triton_swiglu --device xpu --dtype bfloat16 --timer event --peak-tflops 183 --peak-bandwidth-gbps 608
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

For `triton_varlen_flash_attention`, set `load_dump` to `true` to load packed inputs from `flash_attention_args.pt` in the repository root. Otherwise, configure `batch`, `num_heads`, `head_dim`, `total_q`, `total_k`, `max_seqlen_q`, and `max_seqlen_k` to generate synthetic packed inputs.

## Measurement flow

By default, the harness wraps each benchmark callable with `timeit.Timer`, chooses an iteration count using an autorange-style loop, then reports median and mean from repeated per-run timings. The benchmark callables synchronize XPU/CUDA work before returning, so accelerator timings include kernel completion rather than only launch overhead.

When `--timer event` is selected, the harness uses XPU/CUDA events to measure device-side elapsed time and temporarily disables the per-run synchronization inside benchmark callables. This is useful when comparing against profiler or event-based benchmark suites.

When `--timer torch` is selected, the harness follows the PyTorch benchmark recipe pattern:

1. Create benchmark inputs on the selected device.
2. Wrap the benchmark callable with `torch.utils.benchmark.Timer`.
3. Measure with `blocked_autorange`.
4. Emit either a comparison table or JSON summary.

The default shapes in `benchmark_shapes.json` are intentionally moderate so the suite is easy to extend without rewriting the harness.

## Roofline metrics

Each benchmark row includes estimated roofline metrics computed from the configured shape and measured median time:

- `TFLOPS`: estimated floating-point operations per second in TFLOPS.
- `Bandwidth (GB/s)`: estimated tensor memory bandwidth in GB/s.
- `%peak_tflops`: `TFLOPS / --peak-tflops * 100`; the default peak is `183` TFLOPS for BF16.
- `%peak_bw`: `Bandwidth (GB/s) / --peak-bandwidth-gbps * 100`; the default bandwidth is `608` GB/s.
- `AI(F/B)`: arithmetic intensity, computed as `TFLOPS * 1000 / Bandwidth (GB/s)`.
- `roofline_tflops`: best achievable performance under the roofline model, `min(peak_tflops, AI(F/B) * peak_bandwidth_gbps / 1000)`.
- `eff_vs_roofline(%)`: `TFLOPS / roofline_tflops * 100`.
- `bound_hint`: `memory-bound` when `AI(F/B)` is below the ridge point, otherwise `compute-bound`.

The FLOP and byte counts are analytical estimates for comparing benchmark cases consistently. For dump-based varlen FlashAttention, the estimate uses `batch`, `total_q`, `total_k`, `num_heads`, and `head_dim` from `benchmark_shapes.json`.
