"""torch.nn.Module wrappers around VideoDepthAnything that give it a
tflite-friendly, fixed-shape streaming interface: one signature for the
first frame (no cache yet) and one for every later frame (fixed-size cache
in, updated cache out).

Each wrapper is fully self-contained, mirroring the sibling On-Device-RVM
converter's RVMWrapper (see RobustVideoMatting/../wrapper.py): given just the
raw conversion parameters (variant, checkpoint, height, width, input_size,
source, context_len), __init__ resolves the model source, builds the model,
loads the checkpoint, and computes the traced working resolution itself --
nothing needs to be precomputed and injected from outside (no more `model`/
`target_h`/`target_w` constructor arguments).

`source` picks which copy of the model source to trace from, mirroring the
On-Device-RVM converter's model/model_gpu split: "original" imports the
unmodified upstream Video-Depth-Anything/video_depth_anything package
directly (never edited by this toolkit). "gpu" imports
Video-Depth-Anything/video_depth_anything_gpu, a parallel copy meant to be
rewritten for TFLite GPU-delegate compatibility (the same role
RobustVideoMatting/model_gpu plays there) -- currently still a byte-for-byte
copy of the original, pending that work; see README.md.
"""
import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# So this module is importable standalone, regardless of whether convert.py
# has already set up sys.path (and regardless of cwd).
_VDA_DIR = Path(__file__).resolve().parent / "Video-Depth-Anything"
if str(_VDA_DIR) not in sys.path:
    sys.path.insert(0, str(_VDA_DIR))

SOURCE_PACKAGES = {
    "original": "video_depth_anything",
    "gpu": "video_depth_anything_gpu",
}

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

NUM_CACHE_SLOTS = 8  # fixed by architecture: 4 motion modules x 2 attention blocks


def load_source(source: str):
    """Returns (VideoDepthAnything, INFER_LEN, Resize) for the given source."""
    if source not in SOURCE_PACKAGES:
        raise ValueError(f"Unknown source {source!r}; expected one of {list(SOURCE_PACKAGES)}")
    package = SOURCE_PACKAGES[source]
    video_depth_stream = importlib.import_module(f"{package}.video_depth_stream")
    transform = importlib.import_module(f"{package}.util.transform")
    return video_depth_stream.VideoDepthAnything, video_depth_stream.INFER_LEN, transform.Resize


def compute_target_size(height: int, width: int, Resize, input_size: int = 518):
    """Mirror VideoDepthAnything.infer_video_depth_one's dynamic resize so a
    tflite-exported model with fixed input/output shapes still matches the
    resolution the original model would have chosen for this frame size."""
    ratio = max(height, width) / min(height, width)
    if ratio > 1.78:
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14

    resize = Resize(
        width=input_size,
        height=input_size,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
    )
    target_w, target_h = resize.get_size(width, height)
    return target_h, target_w


class DepthWrapperBase(torch.nn.Module):
    """Resolves the model source, builds the model, and computes the traced
    working resolution itself in __init__ -- subclasses only need to
    implement forward()."""

    def __init__(self, variant: str, checkpoint: str, height: int, width: int, input_size: int, source: str, context_len: int):
        super().__init__()
        self.variant = variant
        self.height = height
        self.width = width
        # The base working resolution (before compute_target_size()'s
        # aspect-ratio-preserving adjustment) -- same role as
        # infer_video_depth_one()'s own input_size parameter in the upstream
        # model.
        self.input_size = input_size
        self.source = source
        self.context_len = context_len  # how many past frames' hidden state each step attends to

        VideoDepthAnything, _native_infer_len, Resize = load_source(source)
        self.target_h, self.target_w = compute_target_size(height, width, Resize, input_size)
        self.model = self._build_model(VideoDepthAnything, variant, checkpoint)

        # Bicubic downscale as two constant matmuls, for the "gpu" source
        # only -- see _resize_axis_matrix() and preprocess().
        self._resize_as_matmul = source == "gpu" and (height, width) != (self.target_h, self.target_w)
        if self._resize_as_matmul:
            # Stored as 1x1 conv kernels [c_out, c_in, 1, 1] rather than as
            # plain matrices: a bare matmul of a 2D matrix against the 4D
            # activation lowers to BATCH_MATMUL, which the delegate rejects
            # ("BATCH_MATMUL: Not supported batched mat mul case"). Mapping
            # the resampled axis onto the channel axis turns the exact same
            # contraction into a 1x1 CONV_2D instead -- see preprocess().
            wy = self._resize_axis_matrix(height, self.target_h, vertical=True)
            wx = self._resize_axis_matrix(width, self.target_w, vertical=False)
            self.register_buffer("resize_wy_k", wy[:, :, None, None].contiguous())
            self.register_buffer("resize_wx_k", wx[:, :, None, None].contiguous())

        # video_depth_anything_gpu's DinoVisionTransformer can precompute its
        # positional encoding once here (the traced/inference resolution is
        # fixed for this wrapper instance's whole lifetime) instead of via an
        # F.interpolate traced into every forward call -- see dinov2.py's
        # precompute_pos_encoding(). The unmodified upstream "original"
        # source has no such method, hence the hasattr guard. w/h here match
        # prepare_tokens_with_masks's own (x.shape[2], x.shape[3]) unpacking
        # -- i.e. target_h is passed as "w", target_w as "h".
        if hasattr(self.model.pretrained, "precompute_pos_encoding"):
            self.model.pretrained.precompute_pos_encoding(self.target_h, self.target_w)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @staticmethod
    def _resize_axis_matrix(n_in: int, n_out: int, vertical: bool) -> torch.Tensor:
        """Constant [n_out, n_in] matrix reproducing a bicubic resize along
        one axis, so that resize(x) == Wy @ x @ Wx.T.

        TFLite has no bicubic op, so F.interpolate(mode="bicubic") lowers to
        a data-dependent GATHER_ND (plus BROADCAST_TO/CONCATENATION to build
        the sampling grid) -- none of which the GPU delegate supports, which
        is what forces the whole resize region onto the CPU.

        But the scale factor here is fixed at construction time, which makes
        bicubic a *constant linear operator*, and PyTorch's implementation is
        separable -- so it factors exactly into two fixed matrices. Those
        lower to BATCH_MATMUL, which the delegate does support.

        The matrix is recovered by pushing basis vectors through
        F.interpolate itself rather than reimplementing the cubic kernel, so
        PyTorch's exact weights and boundary handling are captured by
        construction. Measured agreement with F.interpolate is ~3.6e-07.
        """
        if n_in == n_out:
            return torch.eye(n_in)
        eye = torch.eye(n_in)
        if vertical:
            x = eye.t().reshape(n_in, 1, n_in, 1)
            y = F.interpolate(x, size=(n_out, 1), mode="bicubic", align_corners=False)
        else:
            x = eye.reshape(n_in, 1, 1, n_in)
            y = F.interpolate(x, size=(1, n_out), mode="bicubic", align_corners=False)
        return y.reshape(n_in, n_out).t().contiguous()

    @staticmethod
    def _build_model(VideoDepthAnything, variant: str, checkpoint: str):
        model = VideoDepthAnything(**MODEL_CONFIGS[variant])
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def preprocess(self, frame):
        frame = frame / 255.0
        frame = frame.permute(0, 3, 1, 2)  # B,H,W,3 -> B,3,H,W
        # height/width and target_h/target_w are plain Python ints fixed at
        # construction time, so this is a static (trace-time) branch: at the
        # model's native resolution (e.g. 518x924) it's skipped entirely,
        # rather than tracing an F.interpolate that's only *mathematically*
        # a no-op -- bicubic resampling at matching src/dst size doesn't
        # evaluate to bit-exact identity in floating point, it just gets
        # very close, which otherwise shows up as a small but real diff
        # against a reference path that never resizes at all.
        if self._resize_as_matmul:
            # Same bicubic arithmetic as F.interpolate, but expressed as two
            # separable 1x1 convolutions so it stays on the GPU delegate --
            # see _resize_axis_matrix(). Each pass rotates the axis being
            # resampled into the channel position, contracts it against the
            # constant kernel, and rotates it back.
            frame = frame.permute(0, 3, 1, 2)              # [B,C,H,W] -> [B,W,C,H]
            frame = F.conv2d(frame, self.resize_wx_k)      # contract W -> target_w
            frame = frame.permute(0, 2, 3, 1)              # [B,Wt,C,H] -> [B,C,H,Wt]
            frame = frame.permute(0, 2, 1, 3)              # [B,C,H,Wt] -> [B,H,C,Wt]
            frame = F.conv2d(frame, self.resize_wy_k)      # contract H -> target_h
            frame = frame.permute(0, 2, 1, 3)              # [B,Ht,C,Wt] -> [B,C,Ht,Wt]
        elif (self.height, self.width) != (self.target_h, self.target_w):
            frame = F.interpolate(frame, size=(self.target_h, self.target_w), mode="bicubic", align_corners=False)
        frame = (frame - self.mean) / self.std
        return frame.unsqueeze(1)  # B,3,h,w -> B,T=1,3,h,w

    def postprocess(self, depth):
        depth = depth.flatten(0, 1).unsqueeze(1)
        if (self.height, self.width) != (self.target_h, self.target_w):
            depth = F.interpolate(depth, size=(self.height, self.width), mode="bilinear", align_corners=True)
        return depth.permute(0, 2, 3, 1)  # B,1,H,W -> B,H,W,1

    # ---- cache layout at the tflite signature boundary ----
    # The model works in 3D [N, T, C] cache tensors, but the TFLite GPU
    # delegate reads a tensor's *first* dimension as the batch size. A 3D
    # [2442, 7, 192] cache therefore declares batch=2442, which collides with
    # the batch-1 activations it has to be added to, and kills delegate init
    # with "Batch size mismatch, expected 627 but got 1" as soon as both land
    # in the same partition. Carrying the cache across the signature boundary
    # as 4D [1, N, T, C] keeps batch=1 everywhere. Both conversions are a
    # plain unsqueeze/squeeze of a size-1 axis, so they cost nothing
    # numerically.

    @staticmethod
    def _to_cache4d(c):
        return c.unsqueeze(0)  # [N,T,C] -> [1,N,T,C]

    @staticmethod
    def _from_cache4d(c):
        return c.squeeze(0)  # [1,N,T,C] -> [N,T,C], what forward_depth expects


class VDAInitWrapper(DepthWrapperBase):
    """First-frame path: cached_hidden_state_list=None inside forward_depth."""

    def forward(self, frame):
        frame = self.preprocess(frame)
        feat = self.model.forward_features(frame)
        depth, cache_list = self.model.forward_depth(feat, frame.shape, None)

        assert len(cache_list) == NUM_CACHE_SLOTS
        assert cache_list[0].shape[1] == 1  # T=1 straight out of the first frame

        # Warm-up trick: tile the single-frame cache to the full context length,
        # the same way the reference implementation seeds its sliding window.
        # Done as an explicit torch.cat of copies rather than .expand(): the
        # TFLite GPU delegate has no BROADCAST_TO kernel at all, and .expand()
        # traces to exactly that (even followed by .contiguous()), whereas
        # CONCATENATION is supported. context_len is a plain Python int fixed
        # at construction time, so the repeat count is a static trace-time
        # constant, not a dynamic op.
        expanded_caches = [torch.cat([c] * self.context_len, dim=1) for c in cache_list]

        depth = self.postprocess(depth)
        return (depth, *[self._to_cache4d(c) for c in expanded_caches])


class VDAStepWrapper(DepthWrapperBase):
    """Every later frame: a real (T=context_len) cache goes in per slot."""

    def forward(self, frame, h0, h1, h2, h3, h4, h5, h6, h7):
        frame = self.preprocess(frame)
        # h0..h7 arrive as 4D [1, N, T, C]; forward_depth wants 3D [N, T, C].
        cache_list = [self._from_cache4d(h) for h in (h0, h1, h2, h3, h4, h5, h6, h7)]
        feat = self.model.forward_features(frame)
        depth, new_cache = self.model.forward_depth(feat, frame.shape, cache_list)
        depth = self.postprocess(depth)

        updated_cache = []
        for tensor, new in zip(cache_list, new_cache):
            # Keep the two anchor slots, drop the oldest rolling slot, append the newest.
            updated_cache.append(torch.cat([tensor[:, :2], tensor[:, 3:], new], dim=1))

        return (depth, *[self._to_cache4d(c) for c in updated_cache])
