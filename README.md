# depth-anything-mlx

Run [Depth-Anything-V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)
on Apple Silicon through MLX — with **zero real PyTorch anywhere in the
stack**.

## How this actually works

This is **not** a from-scratch MLX reimplementation of Depth-Anything-V2's
architecture. It runs the real, unmodified `transformers.AutoModelForDepthEstimation`
+ `AutoImageProcessor` pipeline — real HF weights, real preprocessing,
real model code — against two things:

- **[torch-mlx](https://github.com/bahaehmimdi/torch-mlx)** — a
  from-scratch `torch`-API-compatible layer backed by `mlx.core`
  instead of PyTorch's own ATen/CPU/CUDA backend. `import torch`
  resolves to this instead of real PyTorch.
- **[a torchvision fork](https://github.com/bahaehmimdi/vision)** —
  `AutoImageProcessor` needs `torchvision.transforms.v2` internally.
  Real torchvision's compiled native extension doesn't build in this
  environment for *any* backend (real PyTorch included — unrelated
  native-ABI issues), so this fork's pure-Python API is used with the
  native-extension-dependent parts stubbed out at the exact points
  where they'd otherwise crash on import.

Getting a real HF model loading and running forward through this path
took 6 real fixes to torch-mlx itself (`torch.ops.load_library`,
`flex_attention.AuxRequest`, `torch._C.DisableTorchFunctionSubclass`,
`torch/types.py`, `Tensor.pin_memory`, `torch._dynamo`, `F.interpolate`'s
`antialias`) plus two import-time stubs for torchvision/torchaudio's
compiled-extension loaders (`depth_anything_mlx/_stubs.py`, mirrored from
`vendor/vision`'s own `tv_ops_stub.py`/`ta_stub.py`). All of that lives in
the two vendored repos below — this package is just the glue that wires
them together for this one model.

**Verified end-to-end**: real pretrained safetensors weights load, a
real forward pass runs, output is the correct shape with no NaNs and
sensible depth values on a real test image.

## Install

```bash
git clone --recurse-submodules https://github.com/bahaehmimdi/depth-anything-mlx.git
cd depth-anything-mlx
pip install -e .
```

(`--recurse-submodules` matters — `vendor/torch-mlx` and `vendor/vision`
are git submodules, not pip dependencies, since neither is set up to be
`pip install`-able as-is: torch-mlx has no packaging metadata yet, and
torchvision's real `pip install` would try to build its native
extension, which fails here regardless of backend.)

## Usage

```python
from depth_anything_mlx import DepthAnythingMLX
from PIL import Image

model = DepthAnythingMLX()             # loads once
depth = model.estimate(Image.open("photo.png"))
depth.save("photo_depth.png")          # grayscale, same size as input
```

Pass `compiled=True` to wrap the forward pass in `mx.compile` (via
`torch.func.functional_call`, since `torch.compile` itself doesn't
exist in torch-mlx — see BENCHMARK_RESULTS.md for how and how much it
actually helps: a small, real, consistently-positive ~2-4%, not a
transformative speedup, since this is a matmul/attention-dominated
model rather than the long elementwise chains `mx.compile` mainly
helps):

```python
model = DepthAnythingMLX(compiled=True)
```

Pass `dtype=mx.float16` to run the whole model in fp16 instead of the
default fp32 — a real, verified **~1.22-1.33x** faster (varies with
image size; see BENCHMARK_RESULTS.md's "fp16 revisited" and "native
preprocessing" sections), correct to the same precision ballpark as
real PyTorch's own fp16 (mean pixel diff ~0.04 on the final 0-255 depth
map, imperceptible). This was ruled out earlier in this project's
history — that finding is stale, from before torch-mlx's fused
attention/layer-norm kernels existed. Preprocessing (resize/rescale/
normalize) is also reimplemented natively in this package rather than
going through `AutoImageProcessor`, so it runs in `dtype` too instead
of being locked to fp32 — this is what closes most of the gap between
"model-only fp16" and "actually as fast as fp16 can make it":

```python
import mlx.core as mx
model = DepthAnythingMLX(dtype=mx.float16)
```

## Benchmark

```bash
python3 benchmark.py --iters 10 --real-torch-device cpu
```

Runs the identical model/weights/input through this repo's torch-mlx
backend and through real, unmodified PyTorch + transformers ("the
standard repo" — plain `pip install torch transformers`, nothing from
this project involved), each in its own subprocess, and reports
mean/median/min/max forward-pass timing for both. See
[`BENCHMARK_RESULTS.md`](./BENCHMARK_RESULTS.md) for a recorded run.

**Five torch-mlx fixes came directly out of profiling this repo**
(Rounds 401-405 — antialiased downsampling, weight-matrix caching,
box-filter pre-reduction for large ratios, fusing that into one op, and
— the biggest single win — using MLX's own fused attention kernel
instead of a hand-composed one). Net result at 108MP: the gap to real
PyTorch's MPS backend narrowed from 2.15x to 1.58-1.66x. Full numbers
and methodology in BENCHMARK_RESULTS.md.

## Known limitations

- `F.interpolate(..., antialias=True)` for actual downsampling (torch-mlx
  Round 401) matches real PyTorch to floating-point precision in the
  interior of an image; there's a real, narrow discrepancy in the last
  ~10px near each edge (boundary-clamping details, not the core
  algorithm) — see torch-mlx's Round 401 commit for the exact numbers.
- torch-mlx's resize used to scale with input image size in a way real
  PyTorch's native kernel doesn't (parity at 480×640 widening to
  ~2.15x slower at 108MP). Three real fixes narrowed this: caching the
  shape-only weight matrix, keeping normalization in `mx.array` space
  instead of numpy, and (torch-mlx Round 403) box-filter pre-reduction
  for ≥8x downsample ratios before the exact antialias matmul — the
  same technique Pillow/mipmapping use for large reductions. Net result
  on the full pipeline: ~6% faster at 48MP, ~3.4% at 108MP (modest,
  since the neural network forward pass dominates total time, not
  preprocessing). Round 403 is a genuine, honestly-measured
  accuracy/speed tradeoff above the 8x threshold — ~0.25% relative
  error on a real photo, ~38% on adversarial random noise (the
  realistic case is what matters for this repo's actual use). See
  BENCHMARK_RESULTS.md's "Optimization audit" and "Closing the
  remaining gap" sections for the full numbers.
- Depth-Anything-V2's depth polarity (near-vs-far convention) hasn't
  been cross-checked against any other depth model here — treat
  `estimate()`'s output as Depth-Anything-V2's own scale, not a
  universal convention.
