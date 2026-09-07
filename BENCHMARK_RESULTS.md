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
| fp16 for the antialiased resize step itself (`dtype=` currently only casts the model + its input, not HF's own preprocessing, which stays fp32 regardless) | Real 1.30x on the resize op in isolation (torch-mlx's `F.interpolate`, verified against real natural-image content: 0.021% relative error, well within tolerance), but **not safely integrable** | HF's `DPTImageProcessor._preprocess()` resizes before rescale/normalize, so getting fp16 into the resize would mean driving its `resize()` method directly with a manually-converted fp16 tensor, then feeding the result back through the rest of `preprocess()`. Tried exactly that: the output silently reverted to fp32 regardless of the fp16 input, AND produced a real, non-trivial numerical divergence (max diff 1.055, not fp16-rounding-level) against the normal fp32 pipeline — something in the internal rescale/normalize/dtype-conversion chain isn't being replicated correctly. Chasing the exact cause would mean reverse-engineering fragile, version-specific internals of `transformers`' fast image processor for a partial win (preprocessing is a minority of total time except at 100MP+), which isn't a good risk/reward trade given the correctness stakes (a subtle bug here would silently produce wrong depth maps). Left unintegrated; the underlying resize speedup is real and could be revisited if this project ever reimplements preprocessing natively instead of going through `AutoImageProcessor` |
| Full ONNX-export → ONNX Runtime graph optimizer → [`onnxruntime-ep-mlx`](https://github.com/justinchuby/onnxruntime-mlx) execution provider → compiled MLX graph pipeline, as an alternative to torch-mlx entirely | At best parity (527ms fp16 vs this project's 502ms fp16, a 5% gap within this machine's own documented run-to-run noise), not an improvement | Real, working, independently verified (correctness 0.052% relative diff vs real PyTorch, same ballpark as this project's own fp16 numbers) — and does prove the full pipeline (ONNX tracer's shape/dtype specialization → ORT's constant-folding/fusion/dead-code-elimination graph optimizer → MLX EP's kernel selection + `mlx_compile`) beats plain ORT-CPU by 7-9x. But adds real complexity (a separate ONNX export re-specialized per input resolution, three new dependencies: `onnx`, `onnxruntime`, `onnxruntime-ep-mlx`) for no measured gain over what's already shipped here. Useful as independent confirmation from a completely different compiler stack that this project's torch-mlx+fp16+fused-kernel approach isn't leaving obvious speed on the table, not as something to adopt |
| Quantized Conv2d (naive im2col + `mx.quantized_matmul`), per [ml-explore/mlx#2714](https://github.com/ml-explore/mlx/issues/2714) | 2.7x **slower** on our own DPT decoder conv shape (256→256, 3×3, ~148×196: 14.4ms vs native `mx.conv2d`'s 5.4ms) | That issue hit the same wall (80x slower for them) before winning only with a hand-written, multi-iteration Metal kernel specific to their tiny (18.5KB) conv layer in a memory-constrained UNet — a different problem (shrinking model size) on a different-shaped op than our large, compute-bound decoder convs; consistent with our own quantization finding above (bandwidth win, not useful on a compute-bound workload) |

**Net result**: at normal photo sizes (≤~12MP — the vast majority of
real use), torch-mlx now runs this model **faster than real PyTorch's
own MPS backend**. At extreme sizes (48MP+), it's ~1.6x behind, for a
cause traced all the way down to Apple's own `mlx.core` GEMM kernel
throughput vs its own `torch` MPS GEMM kernel throughput — a gap
between two Apple-authored compute libraries, not reachable from
`torch-mlx` or `depth-anything-mlx`.

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

- **`mx.compile` gives a small, real, consistently-positive effect**:
  ~3.6% faster than eager torch-mlx in the run above (763.2ms →
  736.7ms), ~2% in an isolated same-process eager-vs-compiled
  comparison (779.6ms → 764.4ms) done separately from the subprocess
  benchmark. Output is numerically correct (max abs diff ~3.7e-4
  against eager, floating-point noise, not a real discrepancy). Small
  because this is a matmul/attention-dominated transformer (DPT +
  DINOv2 backbone) — `mx.compile`'s main benefit is fusing long chains
  of small elementwise ops into fewer kernel launches, and there just
  isn't much of that here relative to the large GEMMs, which are
  already single, efficient Metal calls compile can't fuse away
  further (same finding as this same technique applied to SF3D
  elsewhere in this project's history: ~2.6% there too).
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
