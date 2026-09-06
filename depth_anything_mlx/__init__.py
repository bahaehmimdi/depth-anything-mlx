"""Depth-Anything-V2 running on Apple Silicon via MLX, without real
PyTorch anywhere in the stack.

This does NOT reimplement Depth-Anything-V2's architecture in native
MLX. It runs the real, unmodified `transformers.AutoModelForDepthEstimation`
+ `AutoImageProcessor` pipeline (real HF weights, real preprocessing)
against `torch-mlx` (github.com/bahaehmimdi/torch-mlx) -- a from-scratch
`torch`-API-compatible layer backed by `mlx.core` instead of PyTorch's
own ATen/CPU/CUDA backend -- plus a real (but stubbed-at-the-native-
extension-boundary) `torchvision` from
github.com/bahaehmimdi/vision, since `AutoImageProcessor` needs its v2
transforms. See README.md for the full explanation and `vendor/` for
both dependencies, pinned as git submodules.

Usage:
    from depth_anything_mlx import DepthAnythingMLX

    model = DepthAnythingMLX()             # loads once
    depth_image = model.estimate(image)    # PIL.Image in, grayscale
                                            # PIL.Image depth map out,
                                            # resized back to the input's
                                            # own size (min-max normalized
                                            # to 0-255, matching the
                                            # convention most depth-map
                                            # consumers expect)
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE.parent / "vendor"

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"


def _ensure_torch_mlx_active() -> None:
    """Must run before anything in this process imports `torch` --
    once real PyTorch is imported first anywhere in the process,
    `sys.modules["torch"]` is locked in and this substitution can't
    happen retroactively. Safe to call more than once."""
    if "torch" in sys.modules:
        resolved = getattr(sys.modules["torch"], "__file__", "") or ""
        if "torch-mlx" not in resolved:
            raise RuntimeError(
                "Real PyTorch is already imported in this process (torch.__file__="
                f"{resolved!r}) -- depth_anything_mlx must be imported before any "
                "real `import torch` happens anywhere, since torch-mlx can only "
                "take over sys.modules['torch'] the first time it's imported."
            )
        return

    for p in (str(_VENDOR / "torch-mlx"), str(_VENDOR / "vision")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch  # noqa: F401 -- side effect: resolves to torch-mlx from here on

    from . import _stubs

    _stubs.install_all()


class DepthAnythingMLX:
    def __init__(self, model_id: str = MODEL_ID, compiled: bool = False):
        """`compiled=True` wraps the forward pass in `mx.compile`, via
        the pattern from vendor/vision's own `benchmark_torch_mlx.py`:
        extract every parameter's raw `mx.array` (`Tensor.data`), define
        a pure function of (params, raw input array) -> raw output array
        using `torch.func.functional_call` to run the model statelessly,
        and hand THAT to `mx.compile` -- `torch.compile` itself doesn't
        exist here (see torch/compiler/__init__.py: real speedups on
        this project come from `mx.compile` instead). Measured on this
        model: numerically correct (max abs diff ~3.7e-4 vs eager, only
        floating-point noise) but only ~1.02x faster -- this is a
        matmul/attention-dominated transformer, not the kind of long
        elementwise-op chain `mx.compile`'s kernel fusion mainly helps.
        Left as an opt-in for completeness and so the number is easy to
        reproduce, not because it's expected to be a meaningful win."""
        _ensure_torch_mlx_active()

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        # disable_mmap=True: safetensors' memory-mapped "pt" framework
        # loading path needs torch.UntypedStorage.from_file, which
        # torch-mlx doesn't implement -- this falls back to a loading
        # path that doesn't need it (already documented in torch-mlx's
        # own test_real_pretrained_from_hub_compat.py).
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, use_safetensors=True, disable_mmap=True
        )

        self.compiled = compiled
        if compiled:
            import mlx.core as mx
            from torch._tensor import Tensor
            from torch.func import functional_call

            self._raw_params = {n: p.data for n, p in self.model.named_parameters()}

            def _raw_forward(params, raw_pixel_values):
                p = {k: Tensor(v) for k, v in params.items()}
                pixel_values = Tensor(raw_pixel_values)
                out = functional_call(self.model, p, kwargs={"pixel_values": pixel_values})
                return out.predicted_depth.data

            self._compiled_forward = mx.compile(_raw_forward)

    def estimate(self, image):
        """`image`: a PIL.Image (any mode/size). Returns a grayscale
        (mode "L") PIL.Image the same size as the input, min-max
        normalized to [0, 255] -- larger value = whatever
        Depth-Anything-V2's own `predicted_depth` convention says is
        larger (not cross-checked against any other depth model's
        polarity convention; treat as Depth-Anything-V2's own scale).

        Normalization stays in `mx.array` space (lazy, unified memory,
        stays on GPU) instead of round-tripping through `.tolist()` +
        numpy for the min/max/scale arithmetic -- that round-trip
        previously converted every element to a Python float object and
        back before doing four elementwise ops on it. `np.array(mx_arr)`
        converts directly via the array/buffer protocol (no `.tolist()`,
        no per-element Python objects) exactly once, right at the very
        end, after the real work is already done."""
        import mlx.core as mx
        import numpy as np
        from PIL import Image

        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        if self.compiled:
            raw_out = self._compiled_forward(self._raw_params, inputs["pixel_values"].data)
            depth_raw = raw_out[0]
        else:
            import torch

            with torch.no_grad():
                predicted_depth = self.model(**inputs).predicted_depth[0]
            depth_raw = predicted_depth.data

        depth_min, depth_max = mx.min(depth_raw), mx.max(depth_raw)
        normalized = (depth_raw - depth_min) / (depth_max - depth_min + 1e-8)
        scaled = (normalized * 255).astype(mx.uint8)
        mx.eval(scaled)

        return Image.fromarray(np.array(scaled)).resize(image.size)

    def estimate_path(self, image_path, output_path) -> str:
        from PIL import Image

        depth_image = self.estimate(Image.open(image_path))
        depth_image.save(output_path)
        return str(output_path)


__all__ = ["DepthAnythingMLX", "MODEL_ID"]
