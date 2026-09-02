# Helpers specific to this GPU-delegate-compatible fork of dinov2_layers.
# Not part of the upstream DINOv2 source.

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
