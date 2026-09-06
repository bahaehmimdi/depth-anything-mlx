"""Import-time stubs needed to load real `transformers`/`torchvision`
under torch-mlx (an MLX-backed `torch` shim, not real PyTorch).

Both `torchvision` and `torchaudio` ship compiled native extensions that
this project deliberately doesn't and can't implement (there is no real
`.so`/`.dylib` for an MLX-backed shim to dlopen) -- but their own
`__init__.py`s try to load one anyway, at import time, unconditionally,
even for code paths (like plain image preprocessing) that never call a
single native op. `install_all()` prevents that dead-at-runtime
registration code from ever executing, so the surrounding pure-Python
API (transforms, tv_tensors, model classes, ...) imports and works
normally.

Vendored here (rather than only in vendor/vision) so this package has
everything it needs without depending on exactly how vendor/vision is
laid out -- vendor/vision's own copy (tv_ops_stub.py, ta_stub.py) is the
canonical source these were pulled from.
"""

import importlib.machinery
import sys
import types


def _install_torchvision_stubs() -> None:
    """torchvision's compiled `_C`/`_C_stable` extension (nms/roi_align/
    deform_conv2d/...) isn't built for this environment period (real
    PyTorch CPU/MPS included -- unrelated native-ABI issues, not
    torch-mlx-specific). Two things in torchvision/__init__.py's import
    chain assume it loaded successfully anyway, both entirely about the
    compiled ops themselves (never touched by plain model forward/
    backward that never calls a torchvision op):
      - _meta_registrations.py: meta/fake-tensor shape rules
      - _autograd_registrations.py: autocast/autograd dispatch-key
        formulas (torch._C.DispatchKeySet, ...)
    Pre-registered as empty stand-in modules rather than executed.
    Neither touches torchvision.models/torchvision.transforms/the rest
    of the real, unmodified pure-Python package.
    """
    for name in ("torchvision._meta_registrations", "torchvision._autograd_registrations"):
        sys.modules[name] = types.ModuleType(name)

    # torchvision.ops unconditionally imports _register_onnx_ops at
    # package init (wires up torch.onnx symbolic-tracing rules for the
    # compiled ops) -- a large, separate subsystem for exporting traced
    # graphs, out of scope here and unrelated to any eager forward pass.
    mod = types.ModuleType("torchvision.ops._register_onnx_ops")
    mod._register_custom_op = lambda *a, **k: None
    sys.modules["torchvision.ops._register_onnx_ops"] = mod


def _install_torchaudio_stub() -> None:
    """transformers.processing_utils imports torchaudio unconditionally
    for every processor, not just audio ones. Unlike torchvision's own
    extension loader, torchaudio's deliberately does NOT catch a load
    failure (its own docstring says so explicitly) -- it re-raises,
    which would otherwise crash the whole transformers import for a
    plain image-only model that never touches audio.

    No real use here needs audio decoding, so this pre-registers a fake
    torchaudio module before anything can import the real one. Needs a
    real __spec__ (not just any object in sys.modules) -- transformers
    probes availability via importlib.util.find_spec first, which
    raises ValueError on a spec-less module rather than treating it as
    "not found".
    """
    if "torchaudio" in sys.modules:
        return
    stub = types.ModuleType("torchaudio")
    stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    sys.modules["torchaudio"] = stub


def install_all() -> None:
    _install_torchvision_stubs()
    _install_torchaudio_stub()
