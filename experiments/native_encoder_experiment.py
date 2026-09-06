"""EXPERIMENT, NEGATIVE RESULT -- not used by the actual package.

Native mlx.core reimplementation of the DINOv2-Large encoder used by
Depth-Anything-V2 -- no torch-mlx involved at all, direct mx.array end
to end. Built after profiling showed the encoder is ~82% of total
model-forward time, to test the hypothesis that bypassing torch-mlx's
compatibility shim (Tensor wrapper overhead, Python-level dispatch
through its nn.Module machinery) would be faster than running the same
real HF weights through torch-mlx.

**Result: no meaningful speedup.** Verified numerically correct against
the existing torch-mlx backbone (max diff ~5e-5 across all 4 extracted
stages, floating-point noise from 24 layers of accumulated ops) --
then benchmarked: 556.3ms (this native version) vs 562.0ms (torch-mlx
backbone), a 1.01x difference, i.e. within noise.

This is the useful finding: it PROVES (doesn't just argue) that
torch-mlx's shim overhead was never the bottleneck. The actual
remaining gap vs real PyTorch's MPS backend is in the underlying
mlx.core compute kernels themselves (matmul/GEMM throughput -- measured
separately at ~20-27% slower than PyTorch's MPS GEMM for the same
matmul shape), not in anything reachable by rewriting Python-level
code, shim or otherwise. A full native rewrite of the DPT neck/head
too would very likely show the same non-result, for the same reason,
and wasn't attempted given this encoder result.

kept as a documented negative result (same spirit as
bahaehmimdi/ltx2-compile-experiment), not deleted, since a real,
verified "this doesn't help and here's proof" is worth as much as a
positive result for anyone deciding whether to invest in this
direction later.
"""

from __future__ import annotations

import math

import mlx.core as mx


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps) * weight + bias


def _gelu(x: mx.array) -> mx.array:
    return x * 0.5 * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _linear(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    return x @ weight.T + bias


class NativeDinov2Encoder:
    def __init__(self, weights: dict, num_layers: int = 24, num_heads: int = 16,
                 hidden_size: int = 1024, patch_size: int = 14, eps: float = 1e-6,
                 out_layers: tuple[int, ...] = (5, 12, 18, 24)):
        self.w = weights
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.patch_size = patch_size
        self.eps = eps
        self.out_layers = set(out_layers)
        self.scale = self.head_dim ** -0.5

    def _patch_embed(self, pixel_values_nchw: mx.array) -> mx.array:
        # mx.conv2d wants NHWC input and (out, kh, kw, in) weight.
        x = pixel_values_nchw.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        w = self.w["embeddings.patch_embeddings.projection.weight"]  # (out, in, kh, kw)
        w = w.transpose(0, 2, 3, 1)  # -> (out, kh, kw, in)
        out = mx.conv2d(x, w, stride=self.patch_size, padding=0)
        out = out + self.w["embeddings.patch_embeddings.projection.bias"]
        B, H, W, C = out.shape
        return out.reshape(B, H * W, C), H, W

    def _interpolate_pos_encoding(self, num_patches: int, new_h: int, new_w: int, dim: int) -> mx.array:
        pos = self.w["embeddings.position_embeddings"]  # (1, 1+num_positions, dim)
        num_positions = pos.shape[1] - 1
        if num_patches == num_positions and new_h == new_w:
            return pos
        class_pos = pos[:, :1]
        patch_pos = pos[:, 1:]
        sqrt_n = int(round(num_positions ** 0.5))
        patch_pos = patch_pos.reshape(1, sqrt_n, sqrt_n, dim)

        # This grid is tiny (~37x37) and the cost is negligible (~1ms,
        # measured separately) either way -- reusing torch-mlx's own
        # F.interpolate here rather than re-deriving bicubic
        # interpolation from scratch in raw mx.array code, since that
        # path is already verified against real PyTorch elsewhere in
        # this project. antialias=False matches real Dinov2's own
        # interpolate_pos_encoding call exactly (it never passes
        # antialias=True) -- unrelated to the large-photo antialias
        # downsampling this project fixed elsewhere.
        from torch._tensor import Tensor as _TMTensor
        from torch.nn.functional import interpolate as _tm_interpolate

        patch_pos_nchw = _TMTensor(patch_pos.transpose(0, 3, 1, 2))  # NHWC -> NCHW
        resized = _tm_interpolate(patch_pos_nchw, size=(new_h, new_w), mode="bicubic", align_corners=False)
        patch_pos = resized.data.transpose(0, 2, 3, 1).reshape(1, -1, dim)
        return mx.concatenate([class_pos, patch_pos], axis=1)

    def _attention(self, x: mx.array, i: int) -> mx.array:
        p = f"encoder.layer.{i}.attention.attention."
        q = _linear(x, self.w[p + "query.weight"], self.w[p + "query.bias"])
        k = _linear(x, self.w[p + "key.weight"], self.w[p + "key.bias"])
        v = _linear(x, self.w[p + "value.weight"], self.w[p + "value.bias"])

        B, N, _ = x.shape
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.num_heads * self.head_dim)

        op = "encoder.layer.{}.attention.output.dense.".format(i)
        return _linear(out, self.w[op + "weight"], self.w[op + "bias"])

    def _mlp(self, x: mx.array, i: int) -> mx.array:
        p = f"encoder.layer.{i}.mlp."
        h = _linear(x, self.w[p + "fc1.weight"], self.w[p + "fc1.bias"])
        h = _gelu(h)
        return _linear(h, self.w[p + "fc2.weight"], self.w[p + "fc2.bias"])

    def _layer(self, x: mx.array, i: int) -> mx.array:
        p = f"encoder.layer.{i}."
        h = _layer_norm(x, self.w[p + "norm1.weight"], self.w[p + "norm1.bias"], self.eps)
        h = self._attention(h, i)
        h = h * self.w[p + "layer_scale1.lambda1"]
        x = x + h

        h = _layer_norm(x, self.w[p + "norm2.weight"], self.w[p + "norm2.bias"], self.eps)
        h = self._mlp(h, i)
        h = h * self.w[p + "layer_scale2.lambda1"]
        return x + h

    def __call__(self, pixel_values_nchw: mx.array) -> list[mx.array]:
        patches, gh, gw = self._patch_embed(pixel_values_nchw)
        B, num_patches, dim = patches.shape

        cls = mx.broadcast_to(self.w["embeddings.cls_token"], (B, 1, dim))
        x = mx.concatenate([cls, patches], axis=1)
        x = x + self._interpolate_pos_encoding(num_patches, gh, gw, dim)

        outputs = []
        for i in range(self.num_layers):
            x = self._layer(x, i)
            if (i + 1) in self.out_layers:
                outputs.append(_layer_norm(x, self.w["layernorm.weight"], self.w["layernorm.bias"], self.eps))
        return outputs
