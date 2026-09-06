# Benchmark results

`depth-anything/Depth-Anything-V2-Large-hf`, identical real pretrained
weights, identical 480×640 random test image, `benchmark.py`'s default
10 timed iterations after warmup, each backend in its own subprocess.

**Hardware**: Apple M1 Pro, 32GB unified memory.

## vs real PyTorch, CPU

```
torch-mlx (this repo) (device=mlx (unified memory))
  model load: 4.7s
  forward pass over 10 iters:
    mean:   772.0 ms
    median: 765.1 ms
    min:    760.4 ms
    max:    801.3 ms

real PyTorch (cpu) (device=cpu)
  model load: 1.6s
  forward pass over 10 iters:
    mean:   2107.2 ms
    median: 2094.5 ms
    min:    2046.1 ms
    max:    2237.1 ms

torch-mlx is 2.74x faster (median forward-pass time)
```

## vs real PyTorch, MPS (Apple GPU — the fair comparison on this hardware)

```
torch-mlx (this repo) (device=mlx (unified memory))
  model load: 4.5s
  forward pass over 10 iters:
    mean:   809.7 ms
    median: 799.0 ms
    min:    768.9 ms
    max:    871.0 ms

real PyTorch (mps) (device=mps)
  model load: 2.2s
  forward pass over 10 iters:
    mean:   1031.9 ms
    median: 796.9 ms
    min:    774.3 ms
    max:    1686.0 ms

real PyTorch (mps) is 1.00x faster (median forward-pass time)
```

## Takeaway

Against real PyTorch's CPU backend, torch-mlx is clearly faster
(~2.7x) — expected, since MLX targets the same unified-memory GPU/ANE
hardware MPS does, not the CPU path. **Against real PyTorch's own MPS
backend — the actual best-case baseline on Apple Silicon — the two are
at parity** (median times within ~0.3% of each other on this run).
torch-mlx's higher `max` on this run's MPS comparison (1686ms vs 871ms)
suggests MPS has more per-call variance (likely first-real-call kernel
compilation/caching effects even after warmup), not that either backend
is consistently slower.

This isn't "MLX beats PyTorch" — it's "an MLX-array-backed eager
execution of the same model matches PyTorch's own GPU backend on this
hardware," which is a reasonable outcome for eager `mlx.core` ops
against Metal-optimized `torch` MPS kernels, and says nothing about
`mx.compile`, which wasn't used in this comparison (torch-mlx's Tensor
doesn't currently expose a compiled path for an arbitrary
`transformers` model's forward pass through this project's dispatch
layer — a real forward, not a synthetic microbenchmark, so nothing here
is cherry-picked for a favorable number).

Reproduce with:
```bash
python3 benchmark.py --iters 10 --warmup 2 --real-torch-device mps
```
