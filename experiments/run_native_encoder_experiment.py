"""Reproduces the native-encoder experiment's verification + benchmark.
See native_encoder_experiment.py's own docstring for the result and why
it's a documented negative result, not a shipped feature.

usage: python3 experiments/run_native_encoder_experiment.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from depth_anything_mlx import DepthAnythingMLX, _ensure_torch_mlx_active  # noqa: E402

_ensure_torch_mlx_active()

from native_encoder_experiment import NativeDinov2Encoder  # noqa: E402


def median_of(fn, n=10):
    for _ in range(3):
        fn()
    times = []
    for _ in range(n):
        t0 = time.time()
        fn()
        times.append(time.time() - t0)
    return sorted(times)[n // 2]


def main() -> None:
    import mlx.core as mx
    import numpy as np
    import torch
    from PIL import Image

    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((480, 640, 3)) * 255).astype("uint8"))

    model = DepthAnythingMLX()
    m = model.model
    inputs = model.processor(images=img, return_tensors="pt")
    weights = {n: p.data for n, p in m.backbone.named_parameters()}
    native = NativeDinov2Encoder(weights)
    pixel_values_mx = inputs["pixel_values"].data

    native_outs = native(pixel_values_mx)
    mx.eval(native_outs)

    with torch.no_grad():
        ref_out = m.backbone(inputs["pixel_values"])
    ref_feats = [f.data for f in ref_out.feature_maps]
    mx.eval(ref_feats)

    print("Correctness (native vs torch-mlx backbone, per extracted stage):")
    for i, (a, b) in enumerate(zip(native_outs, ref_feats)):
        diff = mx.abs(a - b)
        print(f"  stage {i}: max diff={float(mx.max(diff)):.6f}  mean diff={float(mx.mean(diff)):.6f}")

    t_native = median_of(lambda: mx.eval(native(pixel_values_mx)))

    def torch_mlx_backbone():
        with torch.no_grad():
            out = m.backbone(inputs["pixel_values"])
        mx.eval([f.data for f in out.feature_maps])

    t_shim = median_of(torch_mlx_backbone)

    print(f"\nnative encoder:     {t_native * 1000:.1f} ms")
    print(f"torch-mlx backbone: {t_shim * 1000:.1f} ms")
    print(f"ratio: {t_shim / t_native:.2f}x (expect ~1.0x -- this is the negative result)")


if __name__ == "__main__":
    main()
