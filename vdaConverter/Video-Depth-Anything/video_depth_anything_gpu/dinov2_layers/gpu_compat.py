# Helpers specific to this GPU-delegate-compatible fork of dinov2_layers.
# Not part of the upstream DINOv2 source.

import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn


def linear_as_conv1x1(linear: nn.Linear, x: Tensor) -> Tensor:
    """Apply an nn.Linear as an exactly-equivalent 1x1 convolution.

    nn.Linear lowers to TFLite's FULLY_CONNECTED, whose contract is to
    flatten leading dimensions -- so its output lands as 2D [N, c_out] even
    when the input was [B, N, c_in]. Whatever consumes that output then sees
    a rank mismatch against the still-3D tensors around it, and the GPU
    delegate rejects it: LayerScale's channel-wise multiply reports
    "MUL: Doesn't support broadcasting - input0: [2443,384], input1:
    [1,1,384]", and residual adds report the ADD equivalent.

    No reshape *after* the Linear can fix this -- the flattening is the op's
    contract, and the converter folds any corrective reshape straight back
    out. CONV_2D has no such flattening, so routing the same arithmetic
    through a 1x1 conv keeps the tensor at rank 4 with batch=1.

    The weights are the caller's own nn.Linear parameters, merely viewed as
    [c_out, c_in, 1, 1], so the module remains an nn.Linear and the
    checkpoint's state_dict keys are unchanged.
    """
    b, n, c_in = x.shape
    c_out = linear.weight.shape[0]
    x4 = x.reshape(1, b * n, 1, c_in).permute(0, 3, 1, 2)  # [1, c_in, b*n, 1]
    y4 = F.conv2d(x4, linear.weight.view(c_out, c_in, 1, 1), linear.bias)  # [1, c_out, b*n, 1]
    return y4.permute(0, 2, 3, 1).reshape(b, n, c_out)


def layer_norm(x: Tensor, norm: nn.LayerNorm) -> Tensor:
    """Apply an nn.LayerNorm via an explicit single-axis (dim=-1) mean/var
    reduction, instead of letting torch.export decompose the module's own
    call into whatever MEAN axis list it picks.

    Confirmed by direct flatbuffer inspection: nn.LayerNorm's default
    decomposition lowers to two MEAN ops with `axis=[0, 2]` on a
    [1, N, C]-shaped input -- reducing the batch axis (0, size 1, harmless)
    together with the channel axis (2) in one call, rather than a plain
    dim=-1 reduction. That axis pattern is where the actual GPU-delegate
    bug lives: XNNPACK (CPU) computes it correctly, but the GPU delegate's
    MEAN kernel silently returns a wrong (too large) variance for this
    non-trailing multi-axis combination -- a correctness bug, not a
    rejected/unsupported op, so it produces no error and no CPU fallback.
    Measured effect: DINOv2 block 0's norm1 output alone already diverges
    from the CPU reference by up to ~14 (its own range is roughly ±3-4),
    and it compounds through all 12 residual blocks (~19 by block 11),
    eventually saturating parts of the motion module to NaN.

    Forcing dim=-1 here produces a single-axis MEAN in the exported graph,
    avoiding that axis combination entirely. Mathematically identical to
    nn.LayerNorm's own formula -- `norm` stays a real nn.LayerNorm (state
    dict keys unchanged), only the forward computation route differs.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + norm.eps) * norm.weight + norm.bias
