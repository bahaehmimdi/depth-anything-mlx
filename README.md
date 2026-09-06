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

## Known limitations

- `F.interpolate(..., antialias=True)` only has its documented
  no-op-when-upsampling behavior implemented in torch-mlx so far;
  genuinely antialiased *downsampling* raises `NotImplementedError`
  rather than silently returning an aliased result.
- Depth-Anything-V2's depth polarity (near-vs-far convention) hasn't
  been cross-checked against any other depth model here — treat
  `estimate()`'s output as Depth-Anything-V2's own scale, not a
  universal convention.
