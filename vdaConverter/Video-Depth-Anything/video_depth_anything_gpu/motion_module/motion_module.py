# This file is originally from AnimateDiff/animatediff/models/motion_module.py at main · guoyww/AnimateDiff
# SPDX-License-Identifier: Apache-2.0 license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file was released under [ Apache-2.0 license], with the full license text available at [https://github.com/guoyww/AnimateDiff?tab=Apache-2.0-1-ov-file#readme].
import torch
import torch.nn.functional as F
from torch import nn

from .attention import CrossAttention, FeedForward, apply_rotary_emb, precompute_freqs_cis

from einops import rearrange, repeat
import math

try:
    import xformers
    import xformers.ops

    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available")
    XFORMERS_AVAILABLE = False


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class TemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        num_attention_blocks               = 2,
        norm_num_groups                    = 32,
        temporal_max_len                   = 32,
        zero_initialize                    = True,
        pos_embedding_type                 = "ape",
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads,
            num_layers=num_transformer_block,
            num_attention_blocks=num_attention_blocks,
            norm_num_groups=norm_num_groups,
            temporal_max_len=temporal_max_len,
            pos_embedding_type=pos_embedding_type,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, encoder_hidden_states, attention_mask=None, cached_hidden_state_list=None):
        hidden_states = input_tensor
        hidden_states, output_hidden_state_list = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask, cached_hidden_state_list)

        output = hidden_states
        return output, output_hidden_state_list  # list of hidden states


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        num_attention_blocks               = 2,
        norm_num_groups                    = 32,
        temporal_max_len                   = 32,
        pos_embedding_type                 = "ape",
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    num_attention_blocks=num_attention_blocks,
                    temporal_max_len=temporal_max_len,
                    pos_embedding_type=pos_embedding_type,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def _group_norm(self, x):
        """self.norm applied with its affine step written out explicitly.

        Calling self.norm(x) directly lowers the per-channel weight and bias
        into a GATHER_ND indexed by a constant arange([C]) -- an identity
        gather whose only purpose is to reshape [C] into [1, C]. The TFLite
        GPU delegate has no GATHER_ND kernel at all, so those 8 nodes
        (4 TemporalModules x weight+bias) fracture the graph around every
        motion module.

        Normalizing without affine parameters and applying the scale/shift
        with an explicit reshape is the same arithmetic via a plain
        broadcast. self.norm remains an nn.GroupNorm, so the checkpoint's
        state_dict keys are untouched.
        """
        y = F.group_norm(x, self.norm.num_groups, None, None, self.norm.eps)
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return y * self.norm.weight.view(shape) + self.norm.bias.view(shape)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cached_hidden_state_list=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        output_hidden_state_list = []

        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self._group_norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim).contiguous()
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        if cached_hidden_state_list is not None:
            n = len(cached_hidden_state_list) // len(self.transformer_blocks)
        else:
            n = 0
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, hidden_state_list = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length, attention_mask=attention_mask,
                                                     cached_hidden_state_list=cached_hidden_state_list[i*n:(i+1)*n] if n else None)
            output_hidden_state_list.extend(hidden_state_list)

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output, output_hidden_state_list


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        num_attention_blocks               = 2,
        temporal_max_len                   = 32,
        pos_embedding_type                 = "ape",
    ):
        super().__init__()

        self.attention_blocks = nn.ModuleList(
            [
                TemporalAttention(
                        query_dim=dim,
                        heads=num_attention_heads,
                        dim_head=attention_head_dim,
                        temporal_max_len=temporal_max_len,
                        pos_embedding_type=pos_embedding_type,
                )
                for i in range(num_attention_blocks)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(dim)
                for i in range(num_attention_blocks)
            ]
        )

        self.ff = FeedForward(dim, dropout=0.0, activation_fn="geglu")
        self.ff_norm = nn.LayerNorm(dim)


    @staticmethod
    def _residual_add(branch, residual):
        """Elementwise add of two tensors that are the same shape in eager,
        forced to a matching rank for TFLite export.

        In eager both operands here are [1, N, C]. The exporter, however,
        lowers the branch's trailing nn.Linear to a FULLY_CONNECTED whose
        output it canonicalizes to 2D [N, C], while the residual stream stays
        3D [1, N, C]. The TFLite GPU delegate's ADD kernel then rejects the
        pair outright ("Doesn't support broadcasting - input0: [2442,192],
        input1: [1,2442,192]") even though it is a plain elementwise add of
        equal-sized tensors.

        Left as a plain add for now: every local reshape formulation has
        been tried against the real exported flatbuffer and all of them
        fail, in one of two ways.

          * reshape(-1, C) and reshape(1, -1, 1, C) both survive lowering
            and *do* make the add same-shape -- but they land as 2D
            [N, C], and a 2D tensor declares batch=N to the delegate. That
            collides with the batch-1 tensors around it and kills delegate
            init outright ("Batch size mismatch, expected 627 but got 1"),
            which is strictly worse than the mismatch itself: the whole
            graph falls back to CPU instead of just these nodes.
          * reshape(1, -1, C) is canonicalized away by the exporter (the
            size-1 leading axis is dropped), silently restoring the original
            mismatch.

        Fixing this properly means stopping the branch from collapsing to 2D
        in the first place -- i.e. giving the trailing nn.Linear a
        rank-preserving lowering (TFLite's FULLY_CONNECTED flattens; CONV_2D
        does not), which is a structural change in attention.py rather than
        a reshape here.
        """
        return branch + residual

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_state_list=None):
        output_hidden_state_list = []
        for i, (attention_block, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            norm_hidden_states = norm(hidden_states)
            residual_hidden_states, output_hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                attention_mask=attention_mask,
                cached_hidden_states=cached_hidden_state_list[i] if cached_hidden_state_list is not None else None,
            )
            hidden_states = self._residual_add(residual_hidden_states, hidden_states)
            output_hidden_state_list.append(output_hidden_states)

        hidden_states = self._residual_add(self.ff(self.ff_norm(hidden_states)), hidden_states)

        output = hidden_states
        return output, output_hidden_state_list


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.,
        max_len = 32
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.dtype)
        return self.dropout(x)

class TemporalAttention(CrossAttention):
    def __init__(
            self,
            temporal_max_len                   = 32,
            pos_embedding_type                 = "ape",
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.pos_embedding_type = pos_embedding_type
        self._use_memory_efficient_attention_xformers = True

        self.pos_encoder = None
        self.freqs_cis = None
        if self.pos_embedding_type == "ape":
            self.pos_encoder = PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.,
                max_len=temporal_max_len
            )

        elif self.pos_embedding_type == "rope":
            self.freqs_cis = precompute_freqs_cis(
                kwargs["query_dim"],
                temporal_max_len
            )

        else:
            raise NotImplementedError

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_states=None):
        # TODO: support cache for these
        assert encoder_hidden_states is None
        assert attention_mask is None

        d = hidden_states.shape[1]
        d_in = 0
        if cached_hidden_states is None:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            input_hidden_states = hidden_states  # (bxd) f c
        else:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
            input_hidden_states = hidden_states
            d_in = cached_hidden_states.shape[1]
            hidden_states = torch.cat([cached_hidden_states, hidden_states], dim=1)

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states[:, d_in:, ...])
        dim = query.shape[-1]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if self.freqs_cis is not None:
            seq_len = query.shape[1]
            freqs_cis = self.freqs_cis[:seq_len].to(query.device)
            query, key = apply_rotary_emb(query, key, freqs_cis)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)


        use_memory_efficient = XFORMERS_AVAILABLE and self._use_memory_efficient_attention_xformers
        if use_memory_efficient and (dim // self.heads) % 8 != 0:
            # print('Warning: the dim {} cannot be divided by 8. Fall into normal attention'.format(dim // self.heads))
            use_memory_efficient = False

        # attention, what we cannot get enough of
        if use_memory_efficient:
            query = self.reshape_heads_to_4d(query)
            key = self.reshape_heads_to_4d(key)
            value = self.reshape_heads_to_4d(value)

            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                raise NotImplementedError
                # hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj, applied as an equivalent 1x1 convolution.
        #
        # nn.Linear lowers to TFLite's FULLY_CONNECTED, whose contract is to
        # flatten leading dimensions -- so its output lands as 2D [N, C].
        # That 2D tensor is then added to the 3D [1, N, C] residual stream in
        # TemporalTransformerBlock, and the GPU delegate's ADD kernel rejects
        # the rank mismatch outright. No reshape *after* this point can fix
        # it: the flattening is the op's contract, and the optimizer folds
        # any corrective reshape straight back out.
        #
        # CONV_2D has no such flattening, so routing the same arithmetic
        # through a 1x1 conv keeps the tensor at rank 4 with batch=1. The
        # weights come from the very same nn.Linear module (just viewed as
        # [c_out, c_in, 1, 1]), so self.to_out[0] stays an nn.Linear and the
        # checkpoint's state_dict keys are unchanged.
        linear = self.to_out[0]
        n, f, c_in = hidden_states.shape
        c_out = linear.weight.shape[0]
        x4 = hidden_states.reshape(1, n * f, 1, c_in).permute(0, 3, 1, 2)  # [1, c_in, n*f, 1]
        y4 = F.conv2d(x4, linear.weight.view(c_out, c_in, 1, 1), linear.bias)  # [1, c_out, n*f, 1]
        hidden_states = y4.permute(0, 2, 3, 1).reshape(n, f, c_out)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states, input_hidden_states
