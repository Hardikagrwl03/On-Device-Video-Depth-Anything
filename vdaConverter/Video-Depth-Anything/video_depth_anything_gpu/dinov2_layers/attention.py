# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor
from torch import nn

from .gpu_compat import linear_as_conv1x1


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        head_dim = C // self.num_heads

        # Upstream packs q/k/v into a single rank-5 tensor
        # (.reshape(B, N, 3, heads, head_dim).permute(2, 0, 3, 1, 4)) and then
        # indexes it. The TFLite GPU delegate tops out at 4D, so every op in
        # that chain is rejected: the rank-5 TRANSPOSE and RESHAPEs outright,
        # and the rank-5 SLICEs force the (single, shared) SLICE opcode to
        # version 5 -- which then disqualifies every *other* SLICE in the
        # graph too, since the delegate's max supported version is 2.
        #
        # Splitting the projection along its channel axis *before* reshaping
        # to the head layout keeps every intermediate at rank <= 4. This is
        # an exact rewrite, not an approximation: reshaping [B, N, 3C] to
        # (3, heads, head_dim) is row-major, so index 0/1/2 of that axis is
        # precisely the first/second/third contiguous C-sized block -- which
        # is what split(C, dim=-1) returns.
        qkv = self.qkv(x)  # [B, N, 3C]
        q, k, v = qkv.split(C, dim=-1)  # 3x [B, N, C]

        def to_heads(t: Tensor) -> Tensor:  # [B, N, C] -> [B, heads, N, head_dim]
            return t.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # Applied as a 1x1 conv so the output keeps rank 4 / batch 1 instead
        # of being flattened to 2D by FULLY_CONNECTED -- otherwise the
        # LayerScale multiply that consumes it is a rank mismatch the GPU
        # delegate rejects. See gpu_compat.linear_as_conv1x1.
        x = linear_as_conv1x1(self.proj, x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        