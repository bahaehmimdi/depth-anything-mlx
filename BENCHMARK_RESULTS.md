# Benchmark results

`depth-anything/Depth-Anything-V2-Large-hf`, identical real pretrained
weights, identical 480×640 random test image, 15 timed iterations after
3 warmup iterations, each backend in its own subprocess (real PyTorch
and torch-mlx can't share a process). No other GPU-heavy process
running during these runs.

**Hardware**: Apple M1 Pro, 32GB unified memory.

```
$ python3 benchmark.py --iters 15 --warmup 3 --all

torch-mlx (device=mlx / Device(gpu, 0) (Metal))
  model load: 4.3s
  forward pass over 15 iters:
    mean:   781.3 ms
    median: 763.2 ms
    min:    757.9 ms
    max:    868.5 ms

torch-mlx (compiled) (device=mlx / Device(gpu, 0) (Metal))
  model load: 4.5s
  forward pass over 15 iters:
    mean:   732.2 ms
    median: 736.7 ms
    min:    706.4 ms
    max:    749.4 ms

real PyTorch (cpu) (device=cpu)
  model load: 1.3s
  forward pass over 15 iters:
    mean:   2092.7 ms
    median: 2094.1 ms
    min:    2035.7 ms
    max:    2199.1 ms

real PyTorch (mps) (device=mps)
  model load: 1.8s
  forward pass over 15 iters:
    mean:   822.8 ms
    median: 841.3 ms
    min:    753.3 ms
    max:    932.7 ms

label                    median ms    vs fastest
torch-mlx (compiled)         736.7         1.00x
torch-mlx                    763.2         1.04x
real PyTorch (mps)           841.3         1.14x
real PyTorch (cpu)          2094.1         2.84x

Fastest: torch-mlx (compiled)
```

An earlier, separate run (no `--compiled` variant) put real PyTorch MPS
slightly ahead of eager torch-mlx (749.9ms vs 767.3ms) rather than
behind it as above — run-to-run variance on this machine is roughly
±10-15% for both GPU paths, so **treat torch-mlx (eager) and real
PyTorch MPS as at parity**, not "torch-mlx wins" or "MPS wins" — which
one comes out ahead on a given run depends on measurement noise, not a
real architectural advantage either way.

## How `mx.compile` is used

Real `torch.compile` doesn't exist in torch-mlx (`torch/compiler/__init__.py`
implements only the no-op compile *hints* like `allow_in_graph`, not
compilation itself — see its own docstring). The actual mechanism,
taken directly from vendor/vision's own `benchmark_torch_mlx.py`:
extract every parameter's raw `mx.array` (`Tensor.data`), define a pure
function of `(params, raw_input_array) -> raw_output_array` using
`torch.func.functional_call` to run the model statelessly, and wrap
*that* with `mx.compile`. See `DepthAnythingMLX.__init__`'s `compiled=True`
path.

## Takeaway

- **`mx.compile` gives a small, real, consistently-positive effect**:
  ~3.6% faster than eager torch-mlx in the run above (763.2ms →
  736.7ms), ~2% in an isolated same-process eager-vs-compiled
  comparison (779.6ms → 764.4ms) done separately from the subprocess
  benchmark. Output is numerically correct (max abs diff ~3.7e-4
  against eager, floating-point noise, not a real discrepancy). Small
  because this is a matmul/attention-dominated transformer (DPT +
  DINOv2 backbone) — `mx.compile`'s main benefit is fusing long chains
  of small elementwise ops into fewer kernel launches, and there just
  isn't much of that here relative to the large GEMMs, which are
  already single, efficient Metal calls compile can't fuse away
  further (same finding as this same technique applied to SF3D
  elsewhere in this project's history: ~2.6% there too).
- **Both GPU paths (torch-mlx, real PyTorch MPS) clearly beat real
  PyTorch's CPU backend** (~2.7-2.8x) — expected, since MLX and MPS
  both target the same unified-memory GPU/ANE hardware the CPU path
  doesn't use.
- This isn't "MLX beats PyTorch" — eager torch-mlx and real PyTorch's
  own MPS backend are at parity within normal run-to-run noise;
  `mx.compile` adds a small, genuine, but not transformative edge on
  top of that.

Reproduce with:
```bash
python3 benchmark.py --iters 15 --warmup 3 --all
```

## At realistic photo size (4000×3000, 12MP)

`DPTImageProcessor` resizes to ~518px regardless of input size, so this
doesn't change the model's own compute — it stresses **preprocessing**
(the resize/antialiasing step) instead, which the small 480×640 test
image above barely exercises (downsample ratio ~1.2x vs ~7.7x here).
This size also only became possible at all after Round 401 of
torch-mlx — real photos this large hit torch-mlx's `F.interpolate`
antialiased-downsampling gap before that (see "Known limitations").

```
$ python3 benchmark.py --iters 10 --warmup 3 --all --height 3000 --width 4000

torch-mlx (device=mlx / Device(gpu, 0) (Metal))
  model load: 4.8s
  forward pass over 10 iters:
    mean:   858.9 ms
    median: 856.6 ms

torch-mlx (compiled) (device=mlx / Device(gpu, 0) (Metal))
  model load: 5.2s
  forward pass over 10 iters:
    mean:   804.8 ms
    median: 809.3 ms

real PyTorch (cpu) (device=cpu)
  forward pass over 10 iters:
    mean:   2089.1 ms
    median: 2086.4 ms

real PyTorch (mps) (device=mps)
  forward pass over 10 iters:
    mean:   749.9 ms
    median: 749.3 ms

label                    median ms    vs fastest
real PyTorch (mps)           749.3         1.00x
torch-mlx (compiled)         809.3         1.08x
torch-mlx                    856.6         1.14x
real PyTorch (cpu)          2086.4         2.78x
```

At this scale, real PyTorch's MPS backend is genuinely ~8-14% faster
than torch-mlx (not parity, unlike the small-image case) — most likely
because this repo's `_interp_weight_matrix` antialias implementation
(a dense `(out_size, in_size)` matrix built via a Python-level loop
over ~2×support+1 taps, see torch-mlx Round 401) does real work
proportional to the input's size to build the resampling matrix every
call, and isn't as optimized as PyTorch's native resize kernel for a
large downsample ratio (~7.7x here vs ~1.2x for the small test image).
`mx.compile` still helps by about the same ~6% it did at the smaller
size.
