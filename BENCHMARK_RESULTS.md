# Benchmark results

`depth-anything/Depth-Anything-V2-Large-hf`, identical real pretrained
weights, identical 480×640 random test image, 15 timed iterations after
3 warmup iterations, each backend in its own subprocess (real PyTorch
and torch-mlx can't share a process). No other GPU-heavy process
running during this run.

**Hardware**: Apple M1 Pro, 32GB unified memory.

```
$ python3 benchmark.py --iters 15 --warmup 3 --all

torch-mlx (device=mlx / Device(gpu, 0) (Metal))
  model load: 4.4s
  forward pass over 15 iters:
    mean:   773.1 ms
    median: 767.3 ms
    min:    761.0 ms
    max:    809.7 ms

real PyTorch (cpu) (device=cpu)
  model load: 1.4s
  forward pass over 15 iters:
    mean:   2044.4 ms
    median: 2044.5 ms
    min:    2022.9 ms
    max:    2076.9 ms

real PyTorch (mps) (device=mps)
  model load: 1.8s
  forward pass over 15 iters:
    mean:   749.6 ms
    median: 749.9 ms
    min:    744.8 ms
    max:    756.1 ms

label                    median ms    vs fastest
real PyTorch (mps)           749.9         1.00x
torch-mlx                    767.3         1.02x
real PyTorch (cpu)          2044.5         2.73x

Fastest: real PyTorch (mps)
```

## Takeaway

- **torch-mlx vs real PyTorch's own MPS backend (the actual best-case
  baseline on Apple Silicon): parity**, torch-mlx within 2% of MPS
  (767.3ms vs 749.9ms median). `mx.default_device()` was confirmed
  `Device(gpu, 0)` before trusting this — a genuine GPU-vs-GPU
  comparison (MLX's own Metal kernels vs PyTorch's MPS Metal kernels),
  not CPU vs GPU.
- **Both GPU paths clearly beat real PyTorch's CPU backend** (~2.7x) —
  expected, since MLX and MPS both target the same unified-memory
  GPU/ANE hardware the CPU path doesn't use.
- This isn't "MLX beats PyTorch" — it's "an MLX-array-backed eager
  execution of the same model matches PyTorch's own GPU backend on
  this hardware." `mx.compile` wasn't used in this comparison (no
  compiled path currently exposed for an arbitrary `transformers`
  model's forward pass through this project's dispatch layer) — this
  is eager `mlx.core` ops against Metal-optimized `torch` MPS kernels,
  a real forward pass end to end, not a synthetic microbenchmark.
- An earlier run (system otherwise busy) showed noisier, higher
  absolute numbers with the same parity conclusion at a different
  ratio (1.06x) — reruns without competing load reproduce this cleaner
  1.02x result consistently.

Reproduce with:
```bash
python3 benchmark.py --iters 15 --warmup 3 --all
```
