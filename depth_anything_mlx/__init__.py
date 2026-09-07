"""Depth-Anything-V2 running on Apple Silicon via MLX, without real
PyTorch anywhere in the stack.

This does NOT reimplement Depth-Anything-V2's architecture in native
MLX. It runs the real, unmodified `transformers.AutoModelForDepthEstimation`
model against `torch-mlx` (github.com/bahaehmimdi/torch-mlx) -- a
from-scratch `torch`-API-compatible layer backed by `mlx.core` instead
of PyTorch's own ATen/CPU/CUDA backend -- plus a real (but stubbed-at-
the-native-extension-boundary) `torchvision` from
github.com/bahaehmimdi/vision, needed by `transformers` itself even
though this package's own preprocessing (see `_native_preprocess`
below) doesn't call into it. Preprocessing (resize/rescale/normalize)
is reimplemented natively here rather than calling `AutoImageProcessor`
directly, so it can run in the model's own `dtype` (see BENCHMARK_RESULTS.md's
"native preprocessing" section for why and the real speedup this gets) --
`AutoImageProcessor` is still loaded and kept around for its config
(target size, mean/std, rescale factor), just not for computing
`pixel_values` itself. See README.md for the full explanation and
`vendor/` for both torch-mlx/vision dependencies, pinned as git
submodules.

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

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE.parent / "vendor"

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"


def _constrain_to_multiple_of(val: float, multiple: int, min_val: int = 0, max_val: int | None = None) -> int:
    """Ported from `transformers.models.dpt.image_processing_dpt.
    get_resize_output_image_size`'s own inner helper -- verified bit-
    identical against it (see Round "native preprocessing" in
    BENCHMARK_RESULTS.md)."""
    x = round(val / multiple) * multiple
    if max_val is not None and x > max_val:
        x = math.floor(val / multiple) * multiple
    if x < min_val:
        x = math.ceil(val / multiple) * multiple
    return x


def _native_preprocess(image, processor, dtype):
    """Reimplements `DPTImageProcessor.__call__`'s pixel_values pipeline
    natively (resize -> rescale -> normalize) instead of calling
    `processor(images=image, return_tensors="pt")`, so the whole chain
    can run in `dtype` (e.g. fp16) instead of being locked to fp32.

    This was originally assumed to be blocked: driving the processor's
    own `resize()`/`rescale_and_normalize()` methods (or a from-scratch
    reimplementation of the same math) with a float32 or fp16 tensor
    consistently diverged from the real `processor(images=...)` output
    by a large, non-rounding-level margin (max diff ~1.06). Traced it
    by monkeypatching `DPTImageProcessor.resize` to print its actual
    inputs during a real call: the reference pipeline resizes the
    still-**uint8** tensor, not a float one -- so the antialiased
    bicubic interpolation's output gets implicitly rounded and clamped
    to whole pixel values in [0, 255] (uint8 can't hold the negative/
    >255 "ringing" overshoot a true float resize produces), losing
    sub-pixel precision *before* rescale/normalize ever run. Replicating
    that quantization step (`.round().clip(0, 255)` right after resize,
    before rescale) reproduces the reference bit-exact (max diff
    ~4.8e-7, pure float rounding). Verified this holds with `dtype=
    mx.float16` too (max diff ~5e-4 vs the fp32 reference, matching this
    project's other fp16 error magnitudes) -- and it's not just
    "equally correct": doing the resize itself in fp16 is a REAL further
    speedup (measured ~1.3-1.8x on the resize+quantize step alone,
    largest at big/108MP-scale images where preprocessing is a bigger
    fraction of total time), since this bypasses `AutoImageProcessor`
    entirely rather than casting only the model + its input.

    Scoped to this class's own fixed assumptions (asserted once at
    __init__ time, not silently ignored): bicubic resample, antialiased,
    rescale+normalize both enabled, no center-crop/padding. If a
    different `model_id` uses a processor configured differently, this
    function is not used at all -- see `DepthAnythingMLX.__init__`."""
    import mlx.core as mx
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.v2.functional as tvF
    from torch._tensor import Tensor

    t = tvF.pil_to_tensor(image)
    t = tvF.to_dtype(t, torch.float32, scale=False).unsqueeze(0)
    if dtype is not None:
        t = Tensor(t.data.astype(dtype))

    ih, iw = t.shape[-2], t.shape[-1]
    oh, ow = processor.size.height, processor.size.width
    sh, sw = oh / ih, ow / iw
    if processor.keep_aspect_ratio:
        if abs(1 - sw) < abs(1 - sh):
            sh = sw
        else:
            sw = sh
    new_h = _constrain_to_multiple_of(sh * ih, processor.ensure_multiple_of)
    new_w = _constrain_to_multiple_of(sw * iw, processor.ensure_multiple_of)

    resized = F.interpolate(t, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
    quantized = resized.round().clip(0, 255)

    target_dtype = dtype if dtype is not None else mx.float32
    mean = Tensor(mx.array(processor.image_mean, dtype=target_dtype).reshape(1, 3, 1, 1))
    std = Tensor(mx.array(processor.image_std, dtype=target_dtype).reshape(1, 3, 1, 1))
    return (quantized * processor.rescale_factor - mean) / std


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
    def __init__(self, model_id: str = MODEL_ID, compiled: bool = False, dtype=None):
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
        reproduce, not because it's expected to be a meaningful win.

        `dtype`: pass `mlx.core.float16` to run the whole model (weights
        and activations) in fp16 instead of the default fp32. This was
        RULED OUT earlier in this project's history (recorded as 0.86x,
        slower) -- that finding is now stale, from before torch-mlx's
        Round 405/406 added fused `mx.fast.scaled_dot_product_attention`/
        `mx.fast.layer_norm` kernels. Those fused kernels internally
        preserve the input dtype end-to-end rather than routing through
        the composed Python-level arithmetic that used to force spurious
        fp32 upcasts (see torch-mlx Round 408's `_wrap_weak` fix for the
        exact mechanism). Re-measured after those changes: fp16 is now
        genuinely ~1.35x FASTER than fp32, verified correct against real
        PyTorch's own `model.half()` output (relative diff in the same
        ballpark as real PyTorch's own fp16 vs fp32 diff, not degraded).
        See BENCHMARK_RESULTS.md's "fp16 revisited" section for the full
        numbers. Default (`dtype=None`) stays fp32, unchanged."""
        _ensure_torch_mlx_active()

        import mlx.core as mx

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        # `_native_preprocess` hardcodes this exact configuration (bicubic
        # antialiased resize, rescale+normalize, no crop/pad) -- assert it
        # rather than silently mis-processing images if a different
        # `model_id` is ever passed with a differently-configured processor.
        assert self.processor.resample == 3, "native preprocess assumes bicubic resample"
        assert self.processor.do_rescale and self.processor.do_normalize
        assert not getattr(self.processor, "do_pad", False)
        assert not getattr(self.processor, "do_center_crop", False)
        # disable_mmap=True: safetensors' memory-mapped "pt" framework
        # loading path needs torch.UntypedStorage.from_file, which
        # torch-mlx doesn't implement -- this falls back to a loading
        # path that doesn't need it (already documented in torch-mlx's
        # own test_real_pretrained_from_hub_compat.py).
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, use_safetensors=True, disable_mmap=True
        )

        self.dtype = dtype
        if dtype is not None:
            for p in self.model.parameters():
                p.data = p.data.astype(dtype)

        self.compiled = compiled
        if compiled:
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
        pixel_values = _native_preprocess(image, self.processor, self.dtype)

        if self.compiled:
            raw_out = self._compiled_forward(self._raw_params, pixel_values.data)
            depth_raw = raw_out[0]
        else:
            import torch

            with torch.no_grad():
                predicted_depth = self.model(pixel_values=pixel_values).predicted_depth[0]
            depth_raw = predicted_depth.data

        depth_min, depth_max = mx.min(depth_raw), mx.max(depth_raw)
        normalized = (depth_raw - depth_min) / (depth_max - depth_min + 1e-8)
        scaled = (normalized * 255).astype(mx.uint8)
        mx.eval(scaled)

        # PIL's `.resize()` default (BICUBIC) is real, measured cost at
        # large photo sizes: 212.7ms for the depth map's upscale back to
        # a 12000x9000 original vs BILINEAR's 152.6ms (~28% cheaper) --
        # found via profiling, was previously the single largest
        # unaccounted-for chunk of estimate()'s own time (not a
        # torch-mlx/model cost at all). BILINEAR is a reasonable
        # default for a low-frequency, already-quantized-to-uint8
        # derived signal like a depth map -- smooth, not blocky like
        # NEAREST (46.6ms, cheapest but visibly blocky), and this
        # doesn't need BICUBIC/LANCZOS's sharper photographic fidelity.
        return Image.fromarray(np.array(scaled)).resize(image.size, Image.BILINEAR)

    def estimate_path(self, image_path, output_path) -> str:
        from PIL import Image

        depth_image = self.estimate(Image.open(image_path))
        depth_image.save(output_path)
        return str(output_path)


__all__ = ["DepthAnythingMLX", "MODEL_ID"]
