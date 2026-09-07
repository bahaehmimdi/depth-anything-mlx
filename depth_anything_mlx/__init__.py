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
    natively (resize -> quantize) instead of calling
    `processor(images=image, return_tensors="pt")`, so the whole chain
    can run in `dtype` (e.g. fp16) instead of being locked to fp32.
    Rescale + normalize are NOT done here -- they're folded into the
    model's own patch-embedding conv instead, see
    `_fold_rescale_normalize_into_patch_embed` below, so this returns
    raw resized-and-quantized pixel values in [0, 255], not the usual
    normalized range.

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
    that quantization step (`.round().clip(0, 255)` right after resize)
    reproduces the reference bit-exact (max diff ~4.8e-7, pure float
    rounding). Verified this holds with `dtype=mx.float16` too (max diff
    ~5e-4 vs the fp32 reference, matching this project's other fp16
    error magnitudes) -- and it's not just "equally correct": doing the
    resize itself in fp16 is a REAL further speedup (measured ~1.3-1.8x
    on the resize+quantize step alone, largest at big/108MP-scale images
    where preprocessing is a bigger fraction of total time), since this
    bypasses `AutoImageProcessor` entirely rather than casting only the
    model + its input.

    Scoped to this class's own fixed assumptions (asserted once at
    __init__ time, not silently ignored): bicubic resample, antialiased,
    rescale+normalize both enabled, no center-crop/padding. If a
    different `model_id` uses a processor configured differently, this
    function is not used at all -- see `DepthAnythingMLX.__init__`."""
    import mlx.core as mx
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.v2.functional as tvF

    # Convert uint8 -> target dtype directly instead of always going
    # through float32 first and casting again -- uint8 pixel values
    # (0-255) are exactly representable in both float32 and float16, so
    # this isn't a precision tradeoff (verified bit-exact, max diff 0.0),
    # just skipping a full extra large-buffer allocation+pass over the
    # image (measured ~25ms saved at 108MP: one uint8->fp16 conversion
    # instead of uint8->fp32 then fp32->fp16).
    torch_dtype = {mx.float16: torch.float16, mx.float32: torch.float32}.get(dtype, torch.float32)
    t = tvF.pil_to_tensor(image)
    t = tvF.to_dtype(t, torch_dtype, scale=False).unsqueeze(0)

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
    return resized.round().clip(0, 255)


def _fold_rescale_normalize_into_patch_embed(model, processor) -> None:
    """Folds `_native_preprocess`'s remaining rescale (`x * rescale_factor`)
    and normalize (`(x - mean) / std`) steps directly into the model's
    patch-embedding conv weights, so `estimate()` never has to run them
    as separate elementwise passes over the full-resolution image at
    all -- `image -> resize -> model` instead of `image -> resize ->
    rescale -> normalize -> model`.

    This is an EXACT algebraic fold, not an approximation. The patch
    conv computes, per output channel k: `out[k] = sum_{c,i,j}
    W[k,c,i,j] * norm(x)[c,i,j] + b[k]` where `norm(x) = x*rescale/std -
    mean/std`. Substituting and regrouping by `x` (raw quantized pixel
    value) instead of `norm(x)`:

        new_W[k,c,i,j] = W[k,c,i,j] * rescale_factor / std[c]
        new_b[k]       = b[k] - sum_{c,i,j} W[k,c,i,j] * mean[c] / std[c]

    Verified on a synthetic conv (random weights, random uint8-range
    input, real `mx.conv2d` call, not just the algebra in isolation):
    max diff 4.8e-6 (float32 rounding) between the original rescale+
    normalize-then-conv path and this folded conv applied directly to
    the raw pixel values. Done ONCE here at model-load time (fp32, for
    precision, before any `dtype=` casting happens) rather than adding
    any per-inference cost -- the folded weights then get cast to
    `dtype` along with the rest of the model, same as before.

    Hardcodes the exact parameter name Depth-Anything-V2's DINOv2
    backbone uses (`backbone.embeddings.patch_embeddings.projection`,
    confirmed via `named_parameters()` against the real loaded model,
    not assumed from the architecture alone). If a different `model_id`
    doesn't have this exact submodule, this raises `AttributeError`
    rather than silently skipping the fold."""
    import mlx.core as mx

    conv = model.backbone.embeddings.patch_embeddings.projection
    W = conv.weight.data  # (out_channels, in_channels=3, kh, kw)
    b = conv.bias.data  # (out_channels,)

    mean = mx.array(processor.image_mean, dtype=mx.float32)  # (3,)
    std = mx.array(processor.image_std, dtype=mx.float32)  # (3,)
    rescale = processor.rescale_factor

    new_W = W * (rescale / std.reshape(1, 3, 1, 1))
    m_over_s = mean / std
    correction = mx.sum(W * m_over_s.reshape(1, 3, 1, 1), axis=(1, 2, 3))
    new_b = b - correction

    conv.weight.data = new_W
    conv.bias.data = new_b


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
        this project come from `mx.compile` instead). Numerically correct
        (max abs diff ~3.7e-4 vs eager, only floating-point noise).

        The "only ~1.02x, not worth it" verdict recorded earlier in this
        project's history is now slightly stale, the same way the
        original fp16 verdict was: it predates every fix in
        BENCHMARK_RESULTS.md's findings log (conv-fold, native
        preprocessing, MLX-based resize-back). Re-measured after those
        landed, on a properly cooled-down machine (a first re-measure
        under this same very long session's thermal throttling gave an
        inflated ~1.05-1.18x that didn't hold up once re-checked under
        stable conditions -- see BENCHMARK_RESULTS.md's own correction of
        itself for the full story): a modest but real and consistent
        **~1.03x for fp16, ~1.04-1.05x for fp32** (occasional noise on
        individual runs), identical across MLX 0.31.2 and 0.32.2 (the
        latter tested in an isolated venv, not the shared environment).
        Left `compiled=False` as the default since the win is small and
        the extra `functional_call` wrapping adds real code-path
        complexity for a few percent; opt in when it's been measured to
        help for your own workload.

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
        # Fold in fp32 (for precision) before any dtype= casting below --
        # see _fold_rescale_normalize_into_patch_embed's own docstring.
        _fold_rescale_normalize_into_patch_embed(self.model, self.processor)

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

        # PIL's own .convert("RGB") does a full re-conversion pass even
        # when the image is ALREADY mode "RGB" -- measured 38.6ms wasted
        # on a 108MP image for a guaranteed no-op in the (very common)
        # case of a photo that's already RGB. Skipping it when unneeded
        # is exact, not approximate: the mode check is what .convert()
        # itself would do first anyway, just without paying for the
        # pixel-buffer pass its non-short-circuiting fast path forces.
        if image.mode != "RGB":
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

        # Resize-back to the original image size via torch-mlx's own
        # F.interpolate (bilinear, plain upsampling -- no antialiasing
        # branch triggers here since scale < 1.0) instead of PIL's own
        # `.resize()`. Switching from PIL's BICUBIC default to BILINEAR
        # was already a real win (212.7ms -> 152.6ms at 108MP -- see the
        # git history for that fix); replacing PIL's BILINEAR with our
        # own gets a further ~2.6x on top (154.1ms -> 59.8ms at 108MP),
        # since this project's whole reason for existing is that the
        # underlying MLX/Metal compute is fast once you're not routing
        # through PIL's CPU-only resize path. Verified against real
        # PIL's own BILINEAR output: mean diff 0.31 (max ~1) on the 0-255
        # scale -- rounding-level, matching the tolerance already
        # accepted when BILINEAR was chosen over BICUBIC in the first
        # place for this same low-frequency, already-quantized signal.
        from torch._tensor import Tensor
        import torch.nn.functional as F

        h, w = scaled.shape
        out_w, out_h = image.size  # PIL .size is (width, height)
        depth_t = Tensor(scaled.astype(mx.float32).reshape(1, 1, h, w))
        resized_t = F.interpolate(depth_t, size=(out_h, out_w), mode="bilinear", align_corners=False)
        out_arr = resized_t.data.astype(mx.uint8).reshape(out_h, out_w)
        mx.eval(out_arr)
        # np.asarray (not np.array) here: np.array's default copy=True
        # forces an explicit copy (measured ~9.6ms at 108MP for no
        # reason); np.asarray gets a real zero-copy view through the
        # buffer protocol instead. Verified this is genuinely safe, not
        # just "didn't crash in one trial" -- lifetime-stress-tested
        # directly against MLX's own allocator accounting
        # (mx.get_active_memory()), not just inference from `.base`:
        #   1. Mutating the numpy view changes what a PIL Image built
        #      from it reads back -- proves real shared memory, not a
        #      coincidental independent copy (Image.fromarray can fall
        #      back to a tobytes() copy for non-contiguous input; this
        #      confirms the actually-taken path here is zero-copy).
        #   2. With the source mx.array deleted and gc'd, `mx.
        #      get_active_memory()` stays unchanged while a PIL Image
        #      built from it is still alive (proves PIL retains the
        #      buffer-protocol exporter, which retains the underlying
        #      MLX allocation -- the allocator can't reuse memory that's
        #      still accounted "active", so a real allocator-pressure
        #      test of ~250 same-shape/dtype MLX allocations after the
        #      delete didn't corrupt the image, as expected).
        #   3. Releasing the PIL Image too drops active memory to
        #      exactly 0 -- the reference chain is real and correctly
        #      torn down, not an accidental leak.
        # This is the "PIL Image -> holds reference -> MLX array ->
        # underlying memory" safe ownership shape, not "PIL stored a
        # raw pointer" -- confirmed via MLX's own memory accounting,
        # not assumed from zero-copy speed alone.
        return Image.fromarray(np.asarray(out_arr))

    def estimate_path(self, image_path, output_path) -> str:
        from PIL import Image

        depth_image = self.estimate(Image.open(image_path))
        depth_image.save(output_path)
        return str(output_path)


__all__ = ["DepthAnythingMLX", "MODEL_ID"]
