"""Runs in its own subprocess (real torch and torch-mlx can't coexist
in one process -- only one can ever occupy sys.modules["torch"]),
timing N forward passes of Depth-Anything-V2 through the identical
transformers.AutoModelForDepthEstimation call, on one of two backends:

    --backend torch-mlx   -- this repo's DepthAnythingMLX (torch-mlx +
                              vendored torchvision, real weights)
    --backend real-torch  -- real, unmodified PyTorch + real torchvision
                              ("the standard repo" -- plain `pip install
                              torch transformers`, nothing from this
                              project involved at all)

Prints one JSON line to stdout: {"backend", "device", "load_s", "times_s"}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _make_test_image(height: int = 480, width: int = 640):
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(0)
    arr = (rng.random((height, width, 3)) * 255).astype("uint8")
    return Image.fromarray(arr)


def run_torch_mlx(iters: int, warmup: int, compiled: bool = False, image_size: tuple[int, int] = (480, 640),
                   dtype: str | None = None) -> dict:
    sys.path.insert(0, str(HERE))
    from depth_anything_mlx import DepthAnythingMLX

    import mlx.core as mx

    mx_dtype = {"fp16": mx.float16, "float16": mx.float16}.get(dtype) if dtype else None

    t0 = time.time()
    model = DepthAnythingMLX(compiled=compiled, dtype=mx_dtype)
    load_s = time.time() - t0

    image = _make_test_image(*image_size)
    for _ in range(warmup):
        model.estimate(image)

    times = []
    for _ in range(iters):
        t0 = time.time()
        model.estimate(image)
        times.append(time.time() - t0)

    device = f"mlx / {mx.default_device()} (Metal)"
    backend = "torch-mlx"
    if compiled:
        backend += " (compiled)"
    if mx_dtype is not None:
        backend += " (fp16)"
    return {"backend": backend, "device": device, "load_s": load_s, "times_s": times}


def run_real_torch(iters: int, warmup: int, device: str, image_size: tuple[int, int] = (480, 640),
                    full_pipeline: bool = False) -> dict:
    """`full_pipeline=False` (the original behavior): times only the bare
    `model(**inputs)` forward call, with preprocessing done ONCE outside
    the loop and no resize-back to the original image size at all. This
    is NOT comparable to `run_torch_mlx`'s `model.estimate(image)` number
    -- that one times pil_to_tensor + resize + forward + normalize + PIL
    resize-back, every iteration. Found via profiling depth-anything-mlx's
    own `estimate()` stage-by-stage at 108MP: pil_to_tensor alone cost
    ~274ms and the PIL resize-back ~182ms -- both real, unavoidable costs
    for any caller who actually wants a depth-map image back, and BOTH
    excluded from the `full_pipeline=False` number below. `full_pipeline=
    True` reproduces `estimate()`'s exact contract (same pil_to_tensor
    call, same processor-driven resize, same min-max normalize, same
    BILINEAR resize-back) so the two backends are actually comparable."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from depth_anything_mlx import MODEL_ID

    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    load_s = time.time() - t0

    image = _make_test_image(*image_size)

    if full_pipeline:
        def _step():
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                depth_raw = model(**inputs).predicted_depth[0]
            if device == "mps":
                torch.mps.synchronize()
            d = depth_raw.cpu().numpy() if device != "cpu" else depth_raw.numpy()
            dmin, dmax = d.min(), d.max()
            normalized = (d - dmin) / (dmax - dmin + 1e-8)
            scaled = (normalized * 255).astype(np.uint8)
            Image.fromarray(scaled).resize(image.size, Image.BILINEAR)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)

        def _step():
            with torch.no_grad():
                model(**inputs)
            if device == "mps":
                torch.mps.synchronize()

    for _ in range(warmup):
        _step()

    times = []
    for _ in range(iters):
        t0 = time.time()
        _step()
        times.append(time.time() - t0)

    backend = "real-torch" + (" (full pipeline)" if full_pipeline else "")
    return {"backend": backend, "device": device, "load_s": load_s, "times_s": times}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch-mlx", "real-torch"], required=True)
    parser.add_argument("--device", default="cpu", help="real-torch only: cpu or mps")
    parser.add_argument("--compiled", action="store_true", help="torch-mlx only: wrap the forward pass in mx.compile")
    parser.add_argument("--dtype", choices=["fp16"], default=None,
                         help="torch-mlx only: run the whole model in fp16 instead of fp32")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--full-pipeline", action="store_true",
                         help="real-torch only: time the full pil_to_tensor+resize+forward+resize-back "
                              "pipeline instead of just the bare model(**inputs) call, so it's actually "
                              "comparable to torch-mlx's estimate()")
    args = parser.parse_args()

    size = (args.height, args.width)
    if args.backend == "torch-mlx":
        result = run_torch_mlx(args.iters, args.warmup, args.compiled, size, args.dtype)
    else:
        result = run_real_torch(args.iters, args.warmup, args.device, size, args.full_pipeline)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
