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


def run_real_torch(iters: int, warmup: int, device: str, image_size: tuple[int, int] = (480, 640)) -> dict:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from depth_anything_mlx import MODEL_ID

    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    load_s = time.time() - t0

    image = _make_test_image(*image_size)
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

    return {"backend": "real-torch", "device": device, "load_s": load_s, "times_s": times}


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
    args = parser.parse_args()

    size = (args.height, args.width)
    if args.backend == "torch-mlx":
        result = run_torch_mlx(args.iters, args.warmup, args.compiled, size, args.dtype)
    else:
        result = run_real_torch(args.iters, args.warmup, args.device, size)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
