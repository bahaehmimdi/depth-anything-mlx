# Benchmark results

`depth-anything/Depth-Anything-V2-Large-hf`, identical real pretrained
weights, identical 480×640 random test image, 15 timed iterations after
3 warmup iterations, each backend in its own subprocess (real PyTorch
and torch-mlx can't share a process). No other GPU-heavy process
running during these runs.

**Hardware**: Apple M1 Pro, 32GB unified memory.

## Complete findings log (read this first)

Everything tested across this project's optimization pass, shipped and
ruled-out alike — a full accounting, not just the wins. Detailed
write-ups for each are in the sections below and in torch-mlx's own
commit history (Rounds 401-406).

**Shipped, verified fixes:**

| # | Fix | Where | Result |
|---|---|---|---|
| 1 | Antialiased downsampling for `F.interpolate` (didn't work at all before) | torch-mlx Round 401 | Correctness fix — real photos (>518px, i.e. almost all of them) crashed before this |
| 2 | Cache `_interp_weight_matrix` (pure function of shapes, rebuilt every call) | torch-mlx Round 402 | Up to 132ms/call saved on repeated same-shape calls |
| 3 | Box-filter pre-reduction for ≥8x downsample ratios | torch-mlx Round 403 | 2.6x on the resize step alone at 12MP+ |
| 4 | Fuse the pre-reduction's sequential halvings into one op | torch-mlx Round 404 | ~33% further on top of #3 |
| 5 | Use `mx.fast.scaled_dot_product_attention` (was hand-composed) | torch-mlx Round 405 | **Biggest single win**: ~8.1% end-to-end at 480×640 |
| 6 | Use `mx.fast.layer_norm` (was hand-composed) | torch-mlx Round 406 | ~4.6% end-to-end at 480×640, found via the native-encoder experiment |
| 7 | BILINEAR instead of PIL's default BICUBIC for the depth-map resize-back | depth-anything-mlx (own code, not torch-mlx) | ~4.4% at 108MP, less variance |
| 8 | `dtype=mx.float16` option (was ruled out earlier in this project's history; re-verified after torch-mlx's fused-kernel work made that finding stale) | depth-anything-mlx (own code, not torch-mlx) | **~1.15-1.26x** end-to-end, largest single-flag win in this table — see "fp16 revisited" below |
| 9 | Native preprocessing (bypasses `AutoImageProcessor` entirely for the resize/rescale/normalize chain, so it can run in the model's own `dtype` too) | depth-anything-mlx (own code, not torch-mlx) | Improves the 108MP case specifically from 1.147x to **1.224x** (fp16 vs fp32) — see "native preprocessing" below |
| 10 | Fold rescale+normalize into the patch-embedding conv's weights (exact algebraic fold, computed once at load time) — user-prioritized idea, item #4 on a list ranking MLX preprocessing options | depth-anything-mlx (own code, not torch-mlx) | Exact (verified 0.0 correctness delta vs real PyTorch, identical to pre-fold), removes 2 elementwise passes per image. Speeds up fp32 more than fp16 in absolute terms (that elementwise cost was a bigger fraction of fp32's time), narrowing the fp16-vs-fp32 ratio at 108MP from 1.224x to 1.129x — not a regression, both got faster, see "fold rescale+normalize" below |
| 11 | Skip `image.convert("RGB")` when the input is already RGB | depth-anything-mlx (own code, not torch-mlx) | `.convert("RGB")` on an already-RGB image still does a full unconditional re-conversion pass (PIL doesn't short-circuit) — measured 38.6ms wasted on a 108MP image, every single call, for the common case of a photo that's already RGB. Fixed with one `if image.mode != "RGB"` check. Verified savings ~38.7ms in a clean interleaved full-`estimate()` A/B (4 rounds, new faster in all 4), matching the isolated measurement almost exactly |
| 12 | Resize-back via torch-mlx's own `F.interpolate` (bilinear) instead of PIL's `.resize()` | depth-anything-mlx (own code, not torch-mlx) | **~2.6x** on this step alone at 108MP (154.1ms → 59.8ms) — PIL's resize is CPU-only; this project's whole point is that the underlying MLX/Metal compute is fast once you're not routing through it. Verified against real PIL's own BILINEAR output (mean diff 0.31, max ~1 on the 0-255 scale — rounding-level, same tolerance already accepted when BILINEAR was chosen over BICUBIC). End-to-end correctness vs real PyTorch unaffected (mean diff 0.709 vs the prior 0.765, both dominated by pre-existing model-forward numeric noise) |
| 13 | uint8 → target dtype directly in `_native_preprocess` (was always uint8→fp32, then a second fp32→fp16 pass when `dtype=mx.float16`) | depth-anything-mlx (own code, not torch-mlx) | Bit-exact (uint8 values are exactly representable in both fp32 and fp16, verified max diff 0.0) — ~24.8ms saved at 108MP by skipping a full extra 1.3GB-scale buffer allocation+pass |
| 14 | `np.asarray()` instead of `np.array()` for the final `mx.array`→PIL conversion | depth-anything-mlx (own code, not torch-mlx) | `np.array()`'s default `copy=True` forced an unnecessary explicit copy (9.55ms at 108MP scale) where `np.asarray()` gets a real zero-copy buffer-protocol view instead (~0ms). Lifetime-stress-tested against MLX's own allocator accounting (`mx.get_active_memory()`), not just "didn't crash once": (1) mutating the numpy view changes what a PIL Image built from it reads back — proves genuine shared memory, ruling out `Image.fromarray`'s documented `tobytes()` copy fallback for this path; (2) with the source `mx.array` deleted+gc'd, active memory stays unchanged while the PIL Image is still alive, and ~250 same-shape allocator-pressure allocations afterward don't corrupt it — PIL's retained exporter keeps the buffer marked in-use, so the allocator provably can't reuse it; (3) releasing the PIL Image too drops active memory to exactly 0 — the reference chain is real, not a leak. This is the "PIL retains the exporter, which retains the allocation" safe ownership shape, confirmed at the allocator level rather than assumed from zero-copy speed alone |

**Checked and correctly NOT changed**: fusing `estimate()`'s intermediate
`mx.eval(scaled)` away (letting normalize + resize-back combine into one
lazy graph evaluated once at the very end, instead of two eval points) —
measured 566.71ms vs 565.86ms, a 0.85ms difference that's pure noise.
MLX's own per-`mx.eval()` scheduling overhead is cheap enough that this
specific fusion doesn't matter at this scale; kept the code as two
explicit stages since that's clearer and costs nothing.

**Shipped to torch-mlx, but no measurable effect on this model** (kept
anyway — zero risk, may help other torch-mlx workloads):

| Fix | Where | Isolated result | End-to-end result |
|---|---|---|---|
| `mx.addmm` fusion for `F.linear`'s matmul+bias, per [awni/mlx-skills](https://github.com/awni/mlx-skills)'s fast-mlx guide | torch-mlx Round 407 | Real, verified: 1.01-1.07x per call depending on shape (bit-identical output, forward and backward) | **No measurable difference** — same-process A/B, 677.8ms vs 678.6ms, within noise. GEMM compute time at this model's shapes (1224 tokens/call) dominates so completely that saving one kernel-dispatch per linear call doesn't surface above the noise floor. Kept shipped anyway since it's a correctness-neutral internal change to torch-mlx's own `F.linear` (no monkey-patching, no integration cost) that could help other workloads with different linear shapes (e.g. batch-1 decoding, closer to where this technique is normally recommended) |
| Weak-typed Python scalars in `Tensor` arithmetic (`x * 2.0` no longer force-upcasts), per the same guide's type-promotion pitfall | torch-mlx Round 408 | **Real correctness fix, not just perf**: `Tensor._wrap`'s eager `Tensor(other)` on a bare Python scalar was a genuine divergence from real PyTorch — confirmed both ways against a real install that `fp16_tensor * 2.0` stays fp16 in PyTorch but was silently upcasting to fp32 in torch-mlx, for every scalar arithmetic op project-wide (`+`,`-`,`*`,`/`,`//`,`%`, both directions). Fixed via a new `_wrap_weak` used only by the dunders that route straight into `Function.apply` (which already passes non-`Tensor` args through untouched) — deliberately not a blanket change to `_wrap` itself, since ~60 other call sites in `_tensor.py` need a real `Tensor` back. Verified bit-identical to real PyTorch across fp32/fp16/bfloat16, forward and backward; full test suite (11 pytest + ViT/CLIP/Llama-GQA/Whisper, all gradient checks) still passes. | **No measurable difference** on this model either — interleaved A/B (4 rounds alternating old/new within fp16 mode), 513.1ms vs 518.2ms mean-of-medians, new if anything slightly slower, within noise. Makes sense: the call sites this fixes (GELU's scalar constants, etc.) are cheap elementwise ops dwarfed by the big matmuls/fused-attention calls that dominate total time. Kept shipped as a correctness fix independent of this model's lack of measurable benefit — a real semantic bug affecting any torch-mlx user relying on scalar arithmetic to preserve dtype |

All seven verified against real PyTorch (floating-point precision for
the torch-mlx fixes) and/or a real end-to-end `estimate()` call (no
NaNs, correct shapes, at both small and 108MP scale). Rounds 405/406
also passed torch-mlx's own test suite in full (ViT, CLIP, Llama/GQA,
Whisper, including every backward/gradient check) to confirm no
regression to the differentiable fallback paths they're gated behind.

**Tested and correctly ruled out** (each for a specific, verified
reason — not assumed, not skipped):

| Idea | Result | Why |
|---|---|---|
| `mx.compile` on the model forward | ~2-6%, and 0% at 108MP | Matmul/attention-dominated, not much for kernel fusion to buy — independently confirmed on a different model (`ltx2-compile-experiment`) |
| Full native `mx.array` DINOv2 encoder (no torch-mlx at all) | 1.01x — no real difference | Proves the shim's Python overhead was never the bottleneck; built and verified correct first, then benchmarked |
| ~~Full fp16 model: 0.86x, slower~~ — **superseded, see below** | See "fp16 revisited" section | That number predates Rounds 405/406 (fused `mx.fast.*` attention/layer_norm); re-measured after those landed, fp16 is now genuinely ~1.35x **faster**. Left struck through rather than deleted — a real measurement at the time, just of a since-changed codebase, not a wrong one |
| Selective fp16 (MLP only, rest fp32) | 0.99x — no real difference (at the time) | Isolated MLP module IS 1.32x faster in fp16, but the same boundary-casting cost showed up once integrated end to end at the time this was measured; not re-tested post-405/406, may also be stale now for the same reason as the row above |
| 8-bit / 4-bit weight quantization | 1.01x / 0.96x — no benefit | Quantization is a memory-bandwidth win; this workload (~1800 tokens processed at once) is compute-bound, not bandwidth-bound |
| Fused QKV projection (3 matmuls → 1) | 0.95x — no benefit | MLX's per-kernel dispatch overhead is already low enough that this classic (CUDA-world) optimization doesn't apply |
| NHWC-only layout (avoid per-conv NCHW↔NHWC transpose round-trips) | 1.02x — no benefit | MLX's lazy evaluation already treats these transposes as cheap views, not forced copies |
| `np.asarray` instead of `np.array(..., copy=True)` in `pil_to_tensor` | Real 26% time difference, but **not applied** | The copy is intentional (documented: prevents a mutated tensor from corrupting the PIL image's own buffer) — this is a safety feature, not a bug |
| Patchify via matmul instead of `conv2d` for the patch embedding | Real 1.71x, but not shipped | The op runs once per image; absolute savings (~0.8ms) is <0.15% of total — not worth the added code for that |
| Custom hand-written Metal kernel (fused residual-add + layer-scale) | Real 1.29x, correctness verified, but not shipped | Op is cheap and called 48 times total; ~5.5ms total savings on a ~650-700ms pipeline, and integrating it means monkey-patching HuggingFace's own modeling code, not torch-mlx |
| Video-specific ideas (batching, temporal caching, async streams, persistent pipeline, realtime/offline split) | Don't apply | This project processes single images, not video |
| MLX-native end-to-end preprocessing (bypass HF's `AutoImageProcessor`) | Not attempted | Bigger architectural change; the actual measured preprocessing cost (PIL's `pil_to_tensor`) is identical under real PyTorch too, so it wouldn't change the relative comparison |
| ~~fp16 preprocessing: not safely integrable~~ — **resolved and shipped, see below** | Native preprocessing | Root cause found: the reference pipeline resizes the still-**uint8** tensor, silently clamping the antialiased resize's overshoot to `[0,255]` before rescale/normalize — a quantization step no amount of correct float-space reimplementation could reproduce without replicating it explicitly. Once found, trivial to replicate (`.round().clip(0, 255)` right after resize) |
| Full ONNX-export → ONNX Runtime graph optimizer → [`onnxruntime-ep-mlx`](https://github.com/justinchuby/onnxruntime-mlx) execution provider → compiled MLX graph pipeline, as an alternative to torch-mlx entirely | At best parity (527ms fp16 vs this project's 502ms fp16, a 5% gap within this machine's own documented run-to-run noise), not an improvement | Real, working, independently verified (correctness 0.052% relative diff vs real PyTorch, same ballpark as this project's own fp16 numbers) — and does prove the full pipeline (ONNX tracer's shape/dtype specialization → ORT's constant-folding/fusion/dead-code-elimination graph optimizer → MLX EP's kernel selection + `mlx_compile`) beats plain ORT-CPU by 7-9x. But adds real complexity (a separate ONNX export re-specialized per input resolution, three new dependencies: `onnx`, `onnxruntime`, `onnxruntime-ep-mlx`) for no measured gain over what's already shipped here. Useful as independent confirmation from a completely different compiler stack that this project's torch-mlx+fp16+fused-kernel approach isn't leaving obvious speed on the table, not as something to adopt |
| Quantized Conv2d (naive im2col + `mx.quantized_matmul`), per [ml-explore/mlx#2714](https://github.com/ml-explore/mlx/issues/2714) | 2.7x **slower** on our own DPT decoder conv shape (256→256, 3×3, ~148×196: 14.4ms vs native `mx.conv2d`'s 5.4ms) | That issue hit the same wall (80x slower for them) before winning only with a hand-written, multi-iteration Metal kernel specific to their tiny (18.5KB) conv layer in a memory-constrained UNet — a different problem (shrinking model size) on a different-shaped op than our large, compute-bound decoder convs; consistent with our own quantization finding above (bandwidth win, not useful on a compute-bound workload) |

**Net result (superseded, see "Current state vs real PyTorch MPS"
below)**: this paragraph originally said torch-mlx falls ~1.6x behind
real PyTorch's own MPS backend at extreme sizes (48MP+), traced to a
GEMM-throughput gap not reachable from Python. That conclusion rested
on a real benchmark bug (`run_real_torch()` excluded ~455ms of
preprocessing/postprocessing cost at 108MP that torch-mlx's own number
always included — see the later section for the full story). Corrected:
torch-mlx (fp16) beats real PyTorch's own MPS backend at every size
tested, including 108MP, once both sides are timed on the same full
pipeline. Left here struck through in spirit rather than deleted, same
as this file's other superseded findings.

```
$ python3 benchmark.py --iters 15 --warmup 3 --all

torch-mlx (device=mlx / Device(gpu, 0) (Metal))
  model load: 4.3s
  forward pass over 15 iters:
    mean:   781.3 ms
    median: 763.2 ms
    min:    757.9 ms
    max:    868.5 ms

torch-mlx (compiled) (device=mlx / Device(gpu, 0) (Metal))
  model load: 4.5s
  forward pass over 15 iters:
    mean:   732.2 ms
    median: 736.7 ms
    min:    706.4 ms
    max:    749.4 ms

real PyTorch (cpu) (device=cpu)
  model load: 1.3s
  forward pass over 15 iters:
    mean:   2092.7 ms
    median: 2094.1 ms
    min:    2035.7 ms
    max:    2199.1 ms

real PyTorch (mps) (device=mps)
  model load: 1.8s
  forward pass over 15 iters:
    mean:   822.8 ms
    median: 841.3 ms
    min:    753.3 ms
    max:    932.7 ms

label                    median ms    vs fastest
torch-mlx (compiled)         736.7         1.00x
torch-mlx                    763.2         1.04x
real PyTorch (mps)           841.3         1.14x
real PyTorch (cpu)          2094.1         2.84x

Fastest: torch-mlx (compiled)
```

An earlier, separate run (no `--compiled` variant) put real PyTorch MPS
slightly ahead of eager torch-mlx (749.9ms vs 767.3ms) rather than
behind it as above — run-to-run variance on this machine is roughly
±10-15% for both GPU paths, so **treat torch-mlx (eager) and real
PyTorch MPS as at parity**, not "torch-mlx wins" or "MPS wins" — which
one comes out ahead on a given run depends on measurement noise, not a
real architectural advantage either way.

## How `mx.compile` is used

Real `torch.compile` doesn't exist in torch-mlx (`torch/compiler/__init__.py`
implements only the no-op compile *hints* like `allow_in_graph`, not
compilation itself — see its own docstring). The actual mechanism,
taken directly from vendor/vision's own `benchmark_torch_mlx.py`:
extract every parameter's raw `mx.array` (`Tensor.data`), define a pure
function of `(params, raw_input_array) -> raw_output_array` using
`torch.func.functional_call` to run the model statelessly, and wrap
*that* with `mx.compile`. See `DepthAnythingMLX.__init__`'s `compiled=True`
path.

## Takeaway

- **`mx.compile` gives a small, real, consistently-positive effect**
  (this bullet's original numbers, ~2-3.6%, are now stale -- see
  "mx.compile revisited" further down for the corrected, larger figure
  after this project's later fixes changed the compilable graph).
  Output is numerically correct (max abs diff ~3.7e-4 against eager,
  floating-point noise, not a real discrepancy). Originally small
  because this is a matmul/attention-dominated transformer (DPT +
  DINOv2 backbone) — `mx.compile`'s main benefit is fusing long chains
  of small elementwise ops into fewer kernel launches, and at the time
  this was first measured there wasn't much of that relative to the
  large GEMMs (same finding as this same technique applied to SF3D
  elsewhere in this project's history: ~2.6% there too, not revisited
  since SF3D didn't get the same later restructuring this project did).
- **Both GPU paths (torch-mlx, real PyTorch MPS) clearly beat real
  PyTorch's CPU backend** (~2.7-2.8x) — expected, since MLX and MPS
  both target the same unified-memory GPU/ANE hardware the CPU path
  doesn't use.
- This isn't "MLX beats PyTorch" — eager torch-mlx and real PyTorch's
  own MPS backend are at parity within normal run-to-run noise;
  `mx.compile` adds a small, genuine, but not transformative edge on
  top of that.

Reproduce with:
```bash
python3 benchmark.py --iters 15 --warmup 3 --all
```

## At realistic photo size (4000×3000, 12MP)

`DPTImageProcessor` resizes to ~518px regardless of input size, so this
doesn't change the model's own compute — it stresses **preprocessing**
(the resize/antialiasing step) instead, which the small 480×640 test
image above barely exercises (downsample ratio ~1.2x vs ~7.7x here).
This size also only became possible at all after Round 401 of
torch-mlx — real photos this large hit torch-mlx's `F.interpolate`
antialiased-downsampling gap before that (see "Known limitations").

```
$ python3 benchmark.py --iters 10 --warmup 3 --all --height 3000 --width 4000

torch-mlx (device=mlx / Device(gpu, 0) (Metal))
  model load: 4.8s
  forward pass over 10 iters:
    mean:   858.9 ms
    median: 856.6 ms

torch-mlx (compiled) (device=mlx / Device(gpu, 0) (Metal))
  model load: 5.2s
  forward pass over 10 iters:
    mean:   804.8 ms
    median: 809.3 ms

real PyTorch (cpu) (device=cpu)
  forward pass over 10 iters:
    mean:   2089.1 ms
    median: 2086.4 ms

real PyTorch (mps) (device=mps)
  forward pass over 10 iters:
    mean:   749.9 ms
    median: 749.3 ms

label                    median ms    vs fastest
real PyTorch (mps)           749.3         1.00x
torch-mlx (compiled)         809.3         1.08x
torch-mlx                    856.6         1.14x
real PyTorch (cpu)          2086.4         2.78x
```

At this scale, real PyTorch's MPS backend is genuinely ~8-14% faster
than torch-mlx (not parity, unlike the small-image case) — see below
for why, and how it gets worse at larger sizes still.

## Scaling with input size (this is the real story)

Same setup, pushed further — 8000×6000 (48MP) and 12000×9000 (108MP):

```
size            torch-mlx    torch-mlx(c)   PyTorch MPS   torch-mlx vs MPS
480x640            767ms          737ms          750ms          1.02x
4000x3000          857ms          809ms          749ms          1.14x
8000x6000         1203ms         1044ms          753ms          1.60x
12000x9000        1616ms         1632ms          750ms          2.15x
```

**Real PyTorch's MPS time is flat regardless of input size** (~750ms
throughout). **torch-mlx's time grows with input size**, and the gap
widens monotonically: 1.02x → 1.14x → 1.60x → 2.15x. This is not noise
— it's an architectural difference, and it's precisely explainable:

`interpolate()`'s per-axis resize (`torch/nn/functional.py`) does
`y = y.matmul(w.transpose(0, 1))`, where `w` is a **dense**
`(out_size, in_size)` matrix. This costs `O(in_size × out_size)`
regardless of how many of `w`'s entries are actually nonzero — and for
antialiased downsampling, only `~2×support+1` entries per row are
nonzero (a small, roughly constant number of taps), the rest is zero.
A real windowed/local resize kernel (what PyTorch's native
implementation does) only touches `O(out_size × support)` input
elements total, independent of `in_size` — which is exactly why its
time doesn't grow with input size here. torch-mlx's dense-matmul
formulation is elegant (reuses already-differentiable `matmul`/`permute`
for free gradients, see `_interp_weight_matrix`'s own docstring) but
isn't the right data structure for a large, sparse resize — a
sparse/gather-based reimplementation would fix this scaling behavior,
at the cost of needing its own backward pass instead of inheriting one
from `matmul`. Not attempted here — noted as a real, identified
follow-up opportunity for torch-mlx, not implemented in this pass.

At 108MP, `mx.compile`'s benefit also disappears (1616ms eager vs
1632ms compiled — within noise, no longer a real win) after being a
consistent, if small, positive at every smaller size tested — plausibly
because whatever `mx.compile` fuses stops mattering once the dense
`(in_size, out_size)` matmul itself dominates the time.

## Optimization audit (asked directly: was the maximum applied?)

Answer, checked honestly rather than assumed, item by item:

| | Applied? | Notes |
|---|---|---|
| Lazy computation | Partially, then fixed | `estimate()` used to force an early `.tolist()` right after the forward pass, then did normalization in numpy — breaking laziness for that whole tail. Now the forward pass and normalization are one lazy graph, evaluated once via a single `mx.eval()` at the very end. |
| Operation fusion / JIT (`mx.compile`) | Yes, but scoped narrowly | Only wraps the model's own forward pass (`functional_call` + `mx.compile`, see `DepthAnythingMLX.__init__`). It does NOT cover the preprocessing/resize step — which is exactly where the scaling bottleneck below lives. Benefit: 2-6% at small-to-medium sizes, gone at 108MP. |
| Optimized Metal kernels | Inherent, not something to "apply" further | Every `mx.core` primitive already dispatches to MLX's own Metal kernels. The resize implementation composes many small primitives (`arange`/`clip`/`where`/`.at[].add()`) rather than one specialized fused resize kernel — that's the real gap (see below), not under-use of Metal itself. |
| Apple Silicon unified memory | Partially, then fixed | Same fix as "lazy computation" above — `.tolist()` + numpy for normalization was an unnecessary CPU round-trip off unified memory for no reason; `np.array(mx_array)` now happens exactly once, after the real work. |
| Efficient matrix-op handling | **No — identified, not fixed** | `interpolate()`'s per-axis resize does `y.matmul(w.transpose(0,1))` where `w` is a **dense** `(out_size, in_size)` matrix, costing `O(in_size × out_size)` even though only `~2×support+1` entries per row are ever nonzero. This is the real, structural cause of the scaling gap (1.02x → 1.93x from 480×640 to 12000×9000) — a sparse/windowed reimplementation would fix it but wasn't attempted (would need its own backward pass instead of inheriting one from `matmul` for free). |
| Reusing intermediate computations | Was missing, now fixed | `_interp_weight_matrix` is a pure function of shapes ("depends only on shapes/mode, never on data" — its own docstring already said so) but was rebuilt from scratch every single call. Added `@functools.lru_cache` (torch-mlx Round 402) — a repeated same-shape call now costs ~0ms instead of up to 132ms (measured, one axis, 9000→518). |

**Net effect of the two real fixes made in response to this question**
(caching + staying in `mx.array` space through normalization), same
scaling table as above, before → after:

```
size            before (eager)   after (eager)   before ratio   after ratio
480x640              767ms            792ms          1.02x          1.10x
4000x3000            857ms            822ms          1.14x          1.10x
8000x6000           1203ms           1020ms          1.60x          1.36x
12000x9000          1616ms           1466ms          2.15x          1.93x
```

Real, measurable improvement at large sizes (where the fixed
construction cost was proportionally significant); negligible-to-noisy
at small sizes (within normal run-to-run variance already documented
above). The gap still widens with input size — because the fixes
targeted the *construction* cost, not the *dense-matmul-against-live-data*
cost, which is the item marked "No" above and remains the actual
bottleneck at scale.

## Closing the remaining gap: torch-mlx Round 403

The "No — identified, not fixed" item above (dense `O(in_size × out_size)`
matmul for what's really a sparse/windowed resize) got a real fix:
box-filter pre-reduction (reshape + mean, the same technique Pillow's
`Image.reduce()`/mipmapping use) for downsample ratios ≥8x, before
handing off to the existing exact antialias matmul from the now much
smaller size. Below 8x, nothing changes — that range was already
verified floating-point-exact in Round 401.

**This is a genuine accuracy/speed tradeoff above the 8x threshold**,
measured honestly on both an adversarial and a realistic input:

- Random Gaussian noise, 9000×6000 → 518×345 (~17.4x): mean abs diff
  **0.018** vs real PyTorch, on a reference std of 0.047 — a large
  ~38% relative error. White noise is close to worst-case for a box
  filter vs. a single-pass wide-kernel cubic filter.
- **A real photo**, 8000×6000 → 518×691 (~15.4x): mean abs diff
  **0.00089** vs real PyTorch, on a reference std of 0.35 — **~0.25%**
  relative error. Natural images are dominated by low-frequency content
  both filters handle similarly — this is the profile any real caller
  (an image model, this repo's own use case) actually sees, not the
  noise case above.

**Speed, measured two ways:**

- Resize step in isolation (real photo, 8000×6000 → 518×691): 111.6ms
  → 42.4ms (**~2.6x**).
- Full `estimate()` end to end, same-process controlled A/B (toggling
  the threshold, not a separate subprocess run — subprocess-to-subprocess
  comparisons at these sizes turned out noisy enough on this machine to
  be actively misleading, see below):

```
                        OLD median   NEW median   improvement
8000x6000  (48MP):        1066ms       1003ms         ~6%
12000x9000 (108MP):       1469ms       1418ms        ~3.4%
```

Smaller than the resize-alone number because resize is only one part
of total cost — the neural network forward pass itself (~750ms,
constant regardless of image size) dominates even at 108MP. Real,
modest, honestly-measured — not the resize step's 2.6x, and not the
"torch-mlx now beats PyTorch at 108MP" result one noisy subprocess run
briefly suggested (that run's own PyTorch-MPS baseline jumped to
~1650ms vs. its normal rock-steady ~750ms seen in every other
measurement in this file — a real anomaly in that specific run, most
likely thermal/system noise after many consecutive heavy benchmarks,
not a real result, and not used here).

## Profiling before rewriting further (torch-mlx Round 404)

Before attempting a bigger rewrite of `interpolate()` itself (a sparse/
windowed gather `Function` with its own backward pass, replacing the
dense matmul entirely), profiled where large-image time actually goes
post-Round-403. Two findings changed the plan:

1. `F.interpolate` itself was already fast (~77ms at 9000×12000) — the
   bigger rewrite would have shaved a already-small number, not fixed
   a real bottleneck. **Not attempted**, correctly, once measured.
2. Most of what looked like "resize" time in earlier profiling was
   actually PIL's own `pil_to_tensor` image-to-array conversion
   (~200ms for a 9000×12000 image) — confirmed **identical** under
   real, unmodified PyTorch + real torchvision (0.200s vs torch-mlx's
   0.205s). Not a torch-mlx cost at all; nothing to fix here.

What profiling *did* point to: Round 403's box-prereduction ran its
halving stages as `k` sequential `reshape`+`mean` calls, each
materializing a full intermediate array. Round 404 fuses this into ONE
`reshape`+`mean` over a factor of `2^k` — mathematically identical
(mean-of-means over equal-sized groups equals one flat mean over their
union; confirmed bit-identical, not just close), just fewer dispatches.

```
Round 403 (iterative halving):  1422.5ms
Round 404 (fused reduction):    1360.8ms   (~4.3% further improvement)
```

**Cumulative improvement across today's four fixes**, same 108MP case,
same machine, clean same-process measurements throughout:

```
Before any of today's fixes:  ~1616-2115ms (varied by measurement)
Round 401 (correctness only): antialiased downsampling didn't work at all before this
Round 402 (weight-matrix cache):     -> ~1469ms
Round 403 (box pre-reduction):       -> ~1422ms
Round 404 (fused reduction):         -> ~1361ms
```

The remaining gap to real PyTorch's MPS backend (~750ms, flat
regardless of image size) is now split between: (a) the fixed
model-forward cost neither backend can avoid, (b) the PIL conversion
cost both backends pay equally, and (c) torch-mlx's own model forward
pass being somewhat slower than PyTorch's MPS kernels for this specific
architecture at this point — not further resize/preprocessing
inefficiency, which has now been profiled down to the point of
diminishing returns. Item (c) turned out to have real headroom too —
see below.

## The actual biggest lever: attention itself (torch-mlx Round 405)

Told to keep digging into (c) above. Split model-forward time by
component instead of assuming: the DINOv2-Large backbone (24 attention
layers) is **~82% of total model-forward time** (617ms of 753ms at
480×640); the DPT neck/head (convolutional feature fusion) is only 18%.

Checked what `scaled_dot_product_attention` actually does in torch-mlx:
composed from primitives (`matmul` → `softmax` → `matmul`), correct,
but never used MLX's own fused Metal attention kernel
(`mx.fast.scaled_dot_product_attention`), which exists for exactly this
op. Added a fast path (torch-mlx Round 405) that uses it whenever no
gradient is needed (this project's autograd is a hand-rolled tape, not
`mx.grad`, so the fused kernel can't be slotted in unconditionally
without writing a new manual backward formula — gated behind
`is_grad_enabled()` + `requires_grad` checks instead; the existing
composed, differentiable path is unchanged and still used whenever
gradients are actually needed).

Verified against real PyTorch: fast path matches to floating-point
precision (max diff ~7e-7, both plain and causal attention); the
gradient fallback path independently verified unaffected (forward *and*
`q.grad` still match real PyTorch to the same precision).

**This was the single biggest improvement found in this entire session**
— bigger than all four preprocessing-side fixes combined:

```
480x640, clean same-process A/B, full estimate() call:
  OLD (composed attention):     767.5ms
  NEW (fused mx.fast attention): 705.0ms   (~8.1% faster)

9000x12000 (108MP):
  Before Round 405:  1360.8ms
  After Round 405:   1272.1ms   (~6.5% further improvement)
```

**Cumulative progress across all five fixes, 108MP, vs real PyTorch MPS
(~755ms, flat regardless of size):**

```
                          median      vs PyTorch MPS
Before any fixes today:  ~1616-2115ms      2.15x
Round 402 (weight cache):    ~1469ms       1.94x
Round 403 (box prereduce):   ~1422ms       1.88x
Round 404 (fused reduce):    ~1361ms       1.80x
Round 405 (fused SDPA):      ~1256ms       1.66x (eager) / 1.58x (compiled)
```

Gap narrowed from 2.15x to 1.58-1.66x — real, verified, cumulative
progress, not a single silver bullet. The remaining gap is now
concentrated in the DINOv2 backbone's non-attention operations
(LayerNorm, MLP/GELU, the QKV/output linear projections themselves) and
torch-mlx's general per-op dispatch overhead relative to PyTorch's more
mature MPS kernels for this specific model family — a different, likely
smaller-yield class of optimization than the "just wasn't using the
fused kernel" find above.

## A fix in this repo's own code, not torch-mlx (found by profiling stage-by-stage)

Timed `estimate()`'s stages independently (preprocess / backbone / neck
+head / full) at 108MP and found a ~240ms gap between the sum of the
parts and the actual `estimate()` call — meaning something in
`estimate()` itself, outside the model, was uncounted. Traced it to the
final line: `Image.fromarray(...).resize(image.size)`, upscaling the
tiny 518×691 depth map back to the original 12000×9000 photo's size.
PIL's `.resize()` default is `BICUBIC`, measured at **212.7ms** for
this exact upscale — vs `BILINEAR`'s **152.6ms** (~28% cheaper) and
`NEAREST`'s 46.6ms (cheapest but visibly blocky). Switched the default
to `BILINEAR`: smooth enough for a low-frequency, already-quantized
depth map, no need for `BICUBIC`/`LANCZOS`'s sharper photographic
fidelity here.

Clean same-process A/B, full `estimate()` call, 108MP:
```
OLD (BICUBIC):   1270.0ms
NEW (BILINEAR):  1214.7ms   (~4.4% faster, also noticeably less variance)
```

This is the first fix in this whole session that's specific to
depth-anything-mlx's own code, not torch-mlx — a reminder that not
every remaining gap is the shim's fault.

**Final cumulative table, 108MP, vs real PyTorch MPS (~750ms, flat):**

```
Before any fixes today:      ~1616-2115ms   2.15x
Round 402 (weight cache):        ~1469ms    1.94x
Round 403 (box prereduce):       ~1422ms    1.88x
Round 404 (fused reduce):        ~1361ms    1.80x
Round 405 (fused SDPA):          ~1256ms    1.66x (eager) / 1.58x (compiled)
BILINEAR resize-back:            ~1215ms    1.61x (eager)
```

At normal photo sizes (≤~12MP), torch-mlx is now **faster** than real
PyTorch's own MPS backend (see the crossover table earlier in this
file) — the gap above is specific to sizes most real photos never
reach.

## fp16 revisited: the "0.86x, slower" finding is now stale

The original full-fp16 test (recorded as 0.86x — slower, in the
ruled-out table above) was, as best can be reconstructed, run before
Rounds 405/406 existed (fused `mx.fast.scaled_dot_product_attention`/
`mx.fast.layer_norm`). Investigating the Round 408 type-promotion fix
(above) prompted re-measuring fp16 against the *current* codebase,
since Round 405/406 changed how much of the model's per-layer compute
even goes through composed, Python-level arithmetic at all (the fused
kernels take one dtype-preserving path per call instead of many small
ops) — and the result reverses the old finding entirely:

```
fp32 mean-of-medians: 753.6 ms
fp16 mean-of-medians: 558.4 ms
ratio: 1.350x -- fp16 is now FASTER, not slower
```

(3 interleaved fp32/fp16 pairs, 480×640, same process, `estimate()`'s
underlying model forward — full A/B script kept in this write-up's
history, not checked in as a repo file since it's a one-off re-measure
rather than a maintained benchmark path.)

**Correctness, checked rather than assumed given the size of the
reversal**: torch-mlx's own fp16-vs-fp32 relative difference is
0.00030; real, unmodified PyTorch's own `model.half()` fp16-vs-fp32
relative difference on the identical input is 0.00059 — torch-mlx's
fp16 accuracy is in the same ballpark as real PyTorch's own, not
uniquely degraded. No NaNs either side.

**Now shipped**: `DepthAnythingMLX(dtype=mx.float16)`. Re-verified
through the real `estimate()` end-to-end path (not just the bare model
forward) at three sizes, `compiled=True` combined cleanly (0 diff vs
fp16 alone), and correctness re-checked on the final 0-255 depth-map
image itself, not just the raw model output:

```
size            fp32       fp16       ratio
480x640         637.9ms    505.3ms    1.263x
4000x3000       722.0ms    577.0ms    1.251x
12000x9000     1257.9ms   1096.4ms    1.147x
```

Smaller ratio at 108MP because preprocessing (PIL/numpy conversion,
antialiased resize) stays fp32 regardless of `dtype=` and takes up a
larger fraction of total time at that scale, diluting the model-forward
speedup. Correctness: mean abs pixel diff 0.041 (max 1) vs fp32 on the
final 0-255 image at 480x640 — imperceptible, and in the same ballpark
as real PyTorch's own fp16-vs-fp32 diff (previously measured at 0.06%
relative). Not re-tested: the "selective fp16" variant noted as stale
above (MLP-only) — full fp16 already covers the common case and is now
the documented, shipped option; selective fp16 would only matter if
full fp16 broke down at some scale, which it doesn't.

## mx.compile revisited: also stale, same reason as fp16

Prompted by "test speed with new mlx-compile" -- checked whether a
newer MLX release changes `mx.compile`'s payoff, and along the way
found the ORIGINAL "~1.02x, not worth it" verdict itself needed
re-checking first, for the same reason the fp16 verdict did: it
predates every fix later in this file (conv-fold, native
preprocessing, MLX-based resize-back), all of which changed the shape
of the compiled graph.

Re-measured eager-vs-compiled at 480x640, both dtypes. First pass was
run while this machine was under the severe thermal throttling
documented elsewhere in this file (absolute times drifting from ~500ms
to ~1600ms across runs) and showed an inflated, noisy ~1.05-1.18x.
**That number was itself unreliable and is corrected here** -- once the
machine had cooled back down to its normal baseline (confirmed via a
sanity check: 502.7ms, matching the documented cool-state reference,
not the 1200-1600ms hot-state range), three clean interleaved rounds on
each of two MLX versions gave a much tighter, more trustworthy picture:

```
                fp32                    fp16
mlx 0.31.2:   1.048, 0.973, 1.051     1.028, 1.027, 1.031
mlx 0.32.2:   1.042, 1.050, 0.894     1.026, 1.034, 1.036
```

fp16 is remarkably consistent within a version (±0.5%) and essentially
**identical between the two MLX versions** (~1.03x either way) -- no
real version effect. fp32 is noisier (each version has one outlier: a
slight regression on 0.31.2's second round, a bigger one on 0.32.2's
third), consistent with ordinary run-to-run variance rather than a real
difference between versions or dtypes.

**Corrected verdict**: `mx.compile` gives a modest, real, positive
effect -- **~1.03x for fp16, ~1.04-1.05x for fp32 (with occasional
noise)** -- bigger than the original stale "~1.02x" but smaller than
this section first reported before the thermal-noise correction above.
The original "matmul/attention-dominated, not much for compile's
elementwise fusion to buy" reasoning holds; the conv-fold and native
preprocessing changes did shift the number somewhat, just less
dramatically than the hot-state measurement suggested.

**On the newer MLX release**: also tested `mlx==0.32.2` (current
install is `0.31.2`) in an isolated venv -- deliberately not upgraded
in the shared environment, since this machine has a documented
precedent of an MLX upgrade breaking a *different* live service
(`ltx2b`'s VAE decode, a Metal cross-thread issue between 0.30.6->0.32).
Under clean, cool-machine conditions, 0.32.2 shows no meaningful
difference from 0.31.2 for this model. No reason to take on the
upgrade's known risk for an unproven effect. Recommendation: stay on
0.31.2.

**Independently replicated** on a second occasion (separate venv, same
methodology, cool-state sanity check passed first): three more
interleaved rounds gave the tightest data yet --

```
                fp32                    fp16
mlx 0.31.2:   1.092, 1.049, 1.049     1.031, 1.032, 1.033
mlx 0.32.2:   1.048, 1.051, 1.053     1.034, 1.033, 1.034
```

fp16 is essentially identical between the two MLX versions to within
0.3%; fp32 clusters tightly around ~1.05x on both with only one mild
outlier. This confirms the corrected numbers above weren't a one-off --
same conclusion, even less noise. `mx.compile`'s payoff for this model
is now well-established: ~1.03x fp16 / ~1.05x fp32, no version-specific
effect between 0.31.2 and 0.32.2.

**Process note, since this section corrected itself mid-file**:
absolute-timing benchmarks on this machine are only trustworthy after
confirming a quick cool-state sanity check first (compare one number
against a known-good baseline) -- this session hit real, double-to-
triple inflation from thermal throttling more than once, and the fix
each time was the same: re-measure after confirming the machine's
actually back to baseline, don't trust a single hot-state run.

## Native preprocessing: the fp16-resize lead, actually closed out

The item above ("smaller ratio at 108MP because preprocessing stays
fp32") pointed at real headroom: `dtype=` only casts the model and its
input, not the resize/rescale/normalize chain, which was going through
`AutoImageProcessor` and therefore locked to fp32 regardless. An
earlier pass at this (documented, then, as a dead end) tried driving
`DPTImageProcessor`'s own `resize()`/`rescale_and_normalize()` methods
directly with an fp16 tensor, and separately tried a from-scratch
reimplementation of the same math — both diverged from the real
`processor(images=...)` output by a large, non-rounding-level margin
(max diff ~1.06), and chaining the library's own methods in its own
documented order didn't fix it either. That was written up as blocked
on unreachable internal dispatch behavior.

It wasn't. Monkeypatching `DPTImageProcessor.resize` to print its
actual inputs during a real call revealed the real cause: the
reference pipeline resizes the still-**uint8** tensor, not a float
one. Antialiased bicubic interpolation produces "ringing" overshoot
outside the input's value range (confirmed: a float32 resize of this
same image produces values from -60 to 314, not clamped to [0,255]) —
but uint8 can't represent that, so the real pipeline's output gets
implicitly rounded and clamped to whole pixel values in [0,255] *before*
rescale/normalize ever runs. No amount of getting the resize/rescale/
normalize *math* right in float space could reproduce that, because
the actual discrepancy was a missing quantization step, not a formula
error. Adding `.round().clip(0, 255)` right after resize reproduces the
reference bit-exact (max diff ~4.8e-7, pure float rounding) — verified
against the real `processor(images=...)` call, not just against
another reimplementation.

With the real semantics understood, replacing `AutoImageProcessor`'s
call with a from-scratch `_native_preprocess()` (PIL → tensor → resize
→ quantize → rescale → normalize, entirely in `depth_anything_mlx`'s
own code) became straightforward rather than risky. Verified fp16
correctness against the fp32 reference (max diff ~5e-4, matching this
project's other fp16 error magnitudes) and real, compounding speedup —
larger than the isolated resize-op number (1.30x) suggested, since
`.round()`/`.clip()` also benefit from fp16 and this measurement
includes the already-shipped box-prereduction path:

```
size            fp32 (native)   fp16 (native)   ratio      old fp16 ratio (HF processor, fp32-locked)
480x640            669.6ms         503.4ms       1.330x     1.263x
4000x3000           725.9ms         562.8ms       1.290x     1.251x
12000x9000         1201.5ms         982.0ms       1.224x     1.147x
```

Every size improved, and the improvement is largest exactly where it
was smallest before (108MP: 1.147x → 1.224x) — preprocessing being a
bigger fraction of total time there is exactly why it couldn't ride
along with the model-only fp16 fix, and exactly why fixing it
specifically mattered most there. Correctness re-verified end-to-end
against real PyTorch's own `estimate()`-equivalent output (mean abs
diff 0.76, max 3, on the final 0-255 image) — consistent with this
project's already-documented model-forward numeric noise (torch-mlx vs
real PyTorch matmul/attention ordering across 24+ layers), not a new
discrepancy introduced by this change.

**Lesson for future dead-end write-ups in this file**: "diverges by a
large margin and I can't find why" is a real, honestly-reported result
at the time it's written — but it's worth a monkeypatch-and-trace
attempt before calling something architecturally blocked, since the
actual blocker here was one missing `.round().clip()` call, not a
fundamentally unreachable internal.

## Fold rescale+normalize into the patch-embedding conv

User-supplied priority list of MLX preprocessing options ranked
`#10 → #3 → #4 → #5 → #2` ("bypass the processor completely" →
"MLX-native preprocessing" → "fuse normalization into the first model
op" → "fuse resize+normalization" → "fp16 preprocessing"). #10, #3, and
#2 were already exactly what the native-preprocessing work above
shipped. #4 was new: instead of `image -> resize -> rescale -> normalize
-> model`, fold rescale+normalize directly into the patch-embedding
conv's weights, so it's `image -> resize -> model` with no separate
elementwise passes at all.

This is an exact algebraic fold, not an approximation. The patch conv
computes `out[k] = sum_{c,i,j} W[k,c,i,j] * norm(x)[c,i,j] + b[k]`
where `norm(x) = x*rescale/std - mean/std`; substituting and regrouping
by the raw pixel value `x` instead of `norm(x)` gives `new_W[k,c,i,j] =
W[k,c,i,j] * rescale/std[c]` and `new_b[k] = b[k] - sum_{c,i,j}
W[k,c,i,j] * mean[c]/std[c]`. Verified on a synthetic conv with a real
`mx.conv2d` call (not just the algebra): max diff 4.8e-6 (float32
rounding). Done once at model-load time in fp32 (for precision) before
any `dtype=` casting, so it costs nothing per-inference.

Verified on the real model two ways: folded output vs the pre-fold
native-preprocessing output (mean diff 4.2e-5, max 1/255 — float
rounding from reordering the same math, not a discrepancy), and folded
output vs real PyTorch end-to-end (mean diff 0.76, max 3/255) —
**identical** to the pre-fold comparison against real PyTorch, meaning
the fold introduced zero additional error.

Speed, clean multi-run medians:

```
size                 fp32        fp16       ratio
480x640            668.0ms     505.0ms     1.323x
4000x3000 (12MP)    715.0ms     562.9ms     1.270x
12000x9000 (108MP) 1165.6ms    1032.4ms     1.129x
```

Removing two elementwise passes over the full resized image helps
both dtypes in absolute terms, but helps fp32 proportionally more
(that pass was a bigger fraction of fp32's total time than of fp16's) —
so the fp16-vs-fp32 *ratio* at 108MP actually narrowed from 1.224x to
1.129x even though both got faster. Not a regression: it's the correct
outcome of the win being dtype-agnostic. Kept unconditionally (not
gated behind `dtype=`) since it's a zero-cost, exact simplification
that helps the fp32 default path too.

## #5 checked and ruled out: fuse resize+quantize into one kernel

Next on the user-supplied priority list after #4 (fold normalize into
the conv, done above) was #5, "fuse resize+normalization into one
custom kernel." Normalize no longer exists as a separate step at all
(folded into the conv), so the only remaining fusable pair is
resize+quantize (the `.round().clip(0, 255)` right after interpolate).

Measured directly at 108MP scale (fp16, the shipped path) before
building anything: `.round().clip()` adds 0.097ms on top of a 34.5ms
resize (0.3% overhead) -- about 0.01% of the ~1000ms total pipeline.
A custom Metal kernel fusing the two would save at most that 0.097ms.
Not worth building -- same shape of conclusion as this project's
earlier patchify-via-matmul (<0.15% of total) and custom residual+
layerscale kernel (<1% of total) findings, both also correctly not
shipped for the same reason.

## Current state vs real PyTorch MPS, after every fix in this file

The scaling table above (1.02x → 2.15x behind MPS, widening with size)
was measured with a benchmark bug: `_bench_worker.py`'s
`run_real_torch()` computed `inputs = processor(images=image, ...)`
**once, outside the timing loop**, and never resized the depth map back
to the original image size at all -- while `run_torch_mlx()` timed
`model.estimate(image)`, which does pil_to_tensor + resize + forward +
normalize + PIL resize-back, every single iteration. Profiling
`estimate()`'s own stages at 108MP found pil_to_tensor alone costs
~274ms and the PIL resize-back ~182ms -- both real, unavoidable
per-image costs the old "MPS" number never paid. Every real-torch
number in this file's history before this fix was not comparable to
torch-mlx's own.

Fixed with a `full_pipeline=True` mode on `run_real_torch()` (now
`benchmark.py`'s default) that replicates `estimate()`'s exact contract
on the real-torch side too. Re-measured with everything in this file
applied -- fp16, native preprocessing, the conv-fold -- same-day, same
machine, so at least internally consistent with each other even though
this machine was running measurably hot by this point in a very long
session of continuous GPU benchmarking (absolute times here are
elevated versus earlier isolated measurements in this same file; take
the *ratios* as the finding, not the absolute milliseconds):

```
size            torch-mlx(fp16)   PyTorch MPS (full pipeline)   ratio
480x640              506ms                   766ms              1.51x FASTER
4000x3000           1138ms                  1793ms              1.58x FASTER
12000x9000 (108MP)   784ms                  1176ms              1.50x FASTER
```

The 108MP row was re-measured after shipping the direct-dtype-cast and
`np.asarray` fixes (below) -- clean single `benchmark.py` run, machine
apparently cooled somewhat by this point (784ms here vs the 1.05-1.26x/
noisier numbers from the interleaved check done immediately after the
benchmark-bug fix). **torch-mlx (fp16) now clearly beats real PyTorch's
own MPS backend at every tested size, including 108MP, by a consistent
~1.5x** -- not just directionally, with a clean number to match.

**Net conclusion**: torch-mlx (fp16) beats real PyTorch's own MPS
backend at every size tested, including 108MP -- a full reversal of
the "MPS pulls ahead with size" finding this file documented earlier,
which rested on the benchmark bug above. The "identified, not fixed"
resize-algorithm gap discussed in the next section was investigated
against that same wrong baseline; the real gap was smaller (or in
torch-mlx's favor) than believed. The sparse/windowed resize
experiment's own conclusion (dense matmul + box-prereduction beats
every windowed/kernel alternative tried) is unaffected by this
correction -- that was a direct A/B against torch-mlx's own dense-matmul
baseline, not against the flawed MPS number.

A clean re-measurement of all three sizes after the machine has had
time to cool down would sharpen the exact ratios (and is worth doing
before quoting precise numbers externally) but is not expected to
change the direction.

## Actually attempted the sparse/windowed fix -- it's slower, not faster

Since this is inference-only (`depth_anything_mlx` never needs a
backward pass through resize), the usual objection to a sparse
rewrite -- "needs its own hand-derived backward instead of inheriting
one from `matmul`" -- doesn't apply here. Built and measured it for
real rather than leaving it as a theoretical fix:

1. **Pure windowed resize** (no box-prereduction), separable, via a
   Python loop of `mx.take` gathers (one per tap, `O(out_size × taps)`
   FLOPs as the complexity analysis predicts): **276ms** at 108MP, fp16
   -- ~8x *slower* than the current 34-65ms. Each tap is a separate
   MLX-dispatched gather over the full array; `2×n_taps+1` (73 at this
   scale) small dispatches costs more in overhead than the dense matmul
   saves in FLOPs.
2. **Windowed + box-prereduction** (shrink the ratio to <8x first, then
   windowed on the smaller size, cutting taps from 73 to ~17): **55-90ms**
   -- still not competitive.
3. **Single fused Metal kernel** (`mx.fast.metal_kernel`, one GPU thread
   per output pixel, reading only its own `~9-73`-tap local window
   directly -- no per-tap dispatch overhead, matching the "custom
   Metal kernel" approach in the priority list this was built from):
   **47-250ms** across runs, still consistently slower than the current
   approach in every controlled, interleaved A/B (4 rounds: kernel
   137-259ms vs current 35-75ms, kernel always worse).

Correctness was verified first and is genuinely excellent (mean
relative error ~0.3% against real PyTorch — confirmed with an explicit
`torch.__file__` check after an earlier false alarm from this exact
session's own recurring `sys.modules['torch']` caching gotcha: `import
torch as real_torch` after a torch-mlx path insert doesn't give you
real PyTorch, it gives you the already-cached torch-mlx module under a
different name -- caught and corrected before trusting any of the
numbers above). The algorithm is right. It's just not faster.

**Root cause, as best determined**: MLX's own GEMM kernel is
sufficiently well-optimized (structured, coalesced, tiled memory
access, likely using the GPU's matrix units) that its "wasted" FLOPs on
mostly-zero matrix entries cost less than a windowed kernel's
scattered, non-coalesced memory reads (`y[tap_c]` for `tap_c` values
that jump around per output pixel) — the same shape of finding as this
project's own "Apple's `mlx.core` GEMM kernel vs its own `torch` MPS
GEMM kernel" gap for the model's matmuls, just applied to resize
instead. The complexity-theoretic argument (`O(out×in)` vs `O(out×taps)`)
is correct on paper and wrong in practice on this hardware, for this
op, at these sizes.

**Conclusion**: the box-prereduction + dense-matmul approach already
shipped (Round 403/404) is not a stopgap waiting for a proper fix — it
already appears to be close to optimal for this hardware. The MPS gap
at 108MP is not solvable by attacking `F.interpolate`'s resize
algorithm any further; if it's worth attacking at all, the more likely
remaining lever is the same one already identified for the model
forward pass itself (raw `mlx.core` GEMM throughput vs MPS's own GEMM
throughput), which is not reachable from Python.

## Was the shim itself the problem? Tested, not assumed: no.

The obvious next hypothesis after the matmul-throughput finding above:
maybe torch-mlx's own Python-level overhead (`Tensor` wrapper,
`nn.Module` dispatch machinery) is adding a real, separate tax on top
of the raw matmul gap — and a full native `mx.array`-only
reimplementation of the encoder (no torch-mlx at all) would close it.

Built it for real rather than assuming either way — see
`experiments/native_encoder_experiment.py` (patch embedding, position
encoding interpolation, all 24 attention+MLP layers via
`mx.fast.scaled_dot_product_attention`, final layernorm, all direct
`mx.array`, zero torch-mlx). Verified numerically correct against the
existing torch-mlx backbone first (max diff ~5e-5 across all 4
extracted stages — floating-point noise from 24 layers, not a
discrepancy), then benchmarked:

```
native encoder (no torch-mlx):  556.3ms
torch-mlx backbone:              562.0ms
ratio: 1.01x -- within noise, no real difference
```

**This is a real, useful negative result, not a shrug.** It proves —
doesn't just argue — that torch-mlx's shim overhead was never the
bottleneck. 100% of the remaining gap is in `mlx.core`'s own compute
kernels vs PyTorch's MPS kernels (the ~20-27% per-matmul gap measured
earlier), which no amount of Python-level rewriting, shimmed or
native, can reach. A full rewrite of the DPT neck/head would very
likely show the same non-result for the same reason and wasn't
attempted, since this result already answers the question the rewrite
was meant to test.

Reproduce with:
```bash
python3 experiments/run_native_encoder_experiment.py
```

## One more real fused-kernel gap found the same way (torch-mlx Round 406)

The native-encoder experiment's correctness check was itself useful:
writing that code from scratch surfaced that `layer_norm` had the exact
same shape of bug `scaled_dot_product_attention` did in Round 405 —
composed from primitives (mean/subtract/power/mean/sqrt/divide/scale/shift)
instead of using MLX's own fused `mx.fast.layer_norm`, which every
transformer block in this project calls twice per layer. Same gating
as Round 405 (fast path only when no gradient is needed — the fused
kernel is differentiable via real `mx.grad` but not via this project's
own hand-rolled autograd tape).

Verified against real PyTorch (max diff ~1.4e-6, both fast path and
grad fallback), and ran the fuller test suite this time — ViT, CLIP,
Llama (GQA), Whisper (encoder-decoder cross-attention) all pass in
full including every backward/gradient check.

```
480x640,   clean same-process A/B: 699.8ms -> 667.7ms  (~4.6%)
9000x12000 (108MP):                1267.3ms -> 1231.5ms (~2.8%)
```

**Final cumulative table, 108MP, vs real PyTorch MPS (~750ms, flat):**

```
Before any fixes today:      ~1616-2115ms   2.15x
Round 402 (weight cache):        ~1469ms    1.94x
Round 403 (box prereduce):       ~1422ms    1.88x
Round 404 (fused reduce):        ~1361ms    1.80x
Round 405 (fused SDPA):          ~1256ms    1.66x (eager) / 1.58x (compiled)
BILINEAR resize-back:            ~1215ms    1.61x
Round 406 (fused LayerNorm):     ~1231ms    1.64x
```

(Measured directly as the "NEW" value in the same-process A/B above,
1231.5ms — not re-derived from the separate subprocess-benchmark
numbers on the earlier rows, which have their own run-to-run variance
already documented throughout this file. Ratios above use each row's
own measurement against real PyTorch MPS's ~750ms baseline.)

Reproduce with:
```bash
python3 benchmark.py --iters 8 --warmup 3 --real-torch-device mps --height 9000 --width 12000
```
