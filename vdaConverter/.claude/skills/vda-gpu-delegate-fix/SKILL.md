---
name: vda-gpu-delegate-fix
description: Diagnose and fix a TFLite GPU-delegate-unsupported op in Video-Depth-Anything/video_depth_anything_gpu/. Use when a benchmark_gpu.sh log reports unsupported operations, a "TfLiteGpuDelegate Init" failure, ops falling back to CPU, or when asked to make more of the VDA model run on the GPU delegate.
---

# Fixing a GPU-delegate-unsupported op

`Video-Depth-Anything/video_depth_anything_gpu/` is a parallel copy of the
model source that exists to be rewritten for GPU-delegate compatibility.
`video_depth_anything/` is **read-only upstream** and must never be edited.

As of the last port, a `--source gpu` export is **fully delegated** (no
unsupported ops, no CPU fallback). This skill is the recipe used to get
there, for when a future change regresses it or a new variant/resolution
surfaces something new.

## The loop

```bash
# 1. See what the delegate actually rejects
bash benchmark/benchmark_gpu.sh tflite_models/gpu/<model>_init.tflite
```

**Grep the log for more than `ERROR`.** Unsupported ops are reported in two
different wordings and the second contains neither "ERROR" nor "not
supported":

- `<OP>: Operation is not supported.`
- `<OP>: OP is supported, but tensor type/shape isn't compatible.`

Matching only the first will make you conclude ops disappeared when they
didn't. Use `sed -n '/not supported by GPU delegate/,/operations will run/p'`
to capture the whole block.

```bash
# 2. Find WHICH source line produces the op -- parse the flatbuffer
```

Read the exported model with
`tensorflow.lite.python.schema_py_generated`: iterate
`subgraph.Operators(i)`, map `model.OperatorCodes(op.OpcodeIndex())` through
`BuiltinOperator`, and read each tensor's `Name()`. **The debug name encodes
the full PyTorch module path**, e.g.
`.../TemporalTransformerBlock_0/TemporalAttention_0/torch.nn.modules.linear.Linear_0`
-- that names the exact `nn.Module` to edit. Also build a
`tensor -> producing node` map so you can walk backwards from a bad operand
to whatever created it.

```bash
# 3. Edit video_depth_anything_gpu/, then prove it didn't change the math
python compare.py --variant vits          # see vda-compare

# 4. Prove it changed the GRAPH the way you intended -- re-parse the flatbuffer
./scripts/convert.sh --variant vits --source gpu

# 5. Prove it helps on real hardware
bash benchmark/benchmark_gpu.sh tflite_models/gpu/<model>_init.tflite
```

Steps 4 and 5 are both required. Skipping 4 hides no-op "fixes"; skipping 5
hides fixes that trade one blocker for a worse one.

## Pitfalls that cost real time here

- **Eager PyTorch shapes are not evidence about the exported graph.**
  `torch.export` + litert lowering re-ranks aggressively: `nn.Linear`
  becomes `FULLY_CONNECTED` whose output is canonicalized to 2D `[N, C]`,
  and a size-1 *leading* axis is squeezed away. A `LayerScale` fix written
  as `gamma.view((1,)*(x.dim()-1) + (dim,))` was a **complete no-op** --
  in eager `x.dim()` was 3, so it recreated the very `[1,1,384]` shape that
  was already failing, while the real mismatch came from the exporter
  flattening the *other* operand. Always confirm in the flatbuffer.

- **Small repro scripts mislead on this specific problem.** One probe
  emitted *zero* `MUL` ops because constant-folding fused the scale into the
  Linear weights (which doesn't happen in the real block). Another correctly
  reproduced both the baseline mismatch *and* the exporter's squeeze
  behaviour, and still mispredicted the winning fix, because surrounding ops
  change how the graph canonicalizes. Full conversion is the only reliable
  oracle.

- **The delegate reads dimension 0 as the batch size.** Any fix that
  achieves same-shape operands by flattening to 2D `[N, C]` declares
  `batch=N`, collides with the batch-1 tensors around it, and kills delegate
  init entirely (`Batch size mismatch, expected 627 but got 1`). That is
  strictly *worse* than the original mismatch -- the whole graph falls back
  to CPU instead of just those nodes. **Rank-match upward, keeping batch 1.**

- **Clearing one op can expose a worse one.** Both the `DIV` and the
  `Batch size mismatch` failures only surfaced *after* an `ADD` fix grew the
  delegated partition enough for them to matter. A green flatbuffer check is
  not a green benchmark.

- **A shared opcode version gates every instance of that op.** All 48
  `SLICE` nodes shared one `OperatorCode` entry pinned at version 5 (delegate
  max: 2) because of a handful of rank-5 slices. Fixing those dropped the
  entry to v1 and unblocked every *unrelated* slice in the graph at once.

## Worked examples (all in `video_depth_anything_gpu/`)

| Symptom | Root cause | Fix |
|---|---|---|
| `CONCATENATION: Expected a 4D tensor of shape BxHxWxC but got 1x1x384` (**fatal**) | 3D `cls_token` concat in `dinov2.py` | unsqueeze → cat on a new axis → squeeze, so the concat is rank-4 |
| `SLICE: Max version supported: 2. Requested version 5`, rank-5 `TRANSPOSE`/`RESHAPE` | `attention.py` packs qkv into a rank-5 tensor before splitting; delegate caps at 4D | split the projection **before** reshaping to heads -- every intermediate stays ≤4D |
| `MUL: Doesn't support broadcasting - input0: [2443,384], input1: [1,1,384]` | `LayerScale` consumes a `FULLY_CONNECTED` output flattened to 2D | apply `proj`/`fc2` as 1x1 convs (`gpu_compat.linear_as_conv1x1`) -- `CONV_2D` doesn't flatten. **Biggest single win: 102 → 836 delegated nodes** |
| `DIV: No support of few identical inputs` | softmax over a length-1 sequence lowers to `y/y` (init's first frame has no cache) | short-circuit it: softmax of one logit is exactly 1.0, so `probs @ value == value` |
| `ADD: Doesn't support broadcasting - input0: [2442,192], input1: [1,2442,192]` | residual adds against `FULLY_CONNECTED`-flattened branches | apply `to_out[0]` and `FeedForward`'s last projection as 1x1 convs |
| `GATHER_ND: Operation is not supported.` | `nn.GroupNorm`'s affine params indexed by a constant `arange([C])` -- an **identity gather** just to reshape `[C]`→`[1,C]` | normalize with `F.group_norm(..., None, None, eps)`, then apply scale/shift with an explicit reshape |

Two more live in `wrapper.py` (converter-level, not model source): the cache
crosses the signature boundary as 4D `[1, N, T, C]` so batch stays 1, and
cache seeding uses `torch.cat` instead of `.expand()` (the delegate has no
`BROADCAST_TO` kernel at all).

## Keeping edits exact

`linear_as_conv1x1` is the **only** non-exact rewrite, and only above a
reduction length of 384 -- see `vda-compare` for the tolerance rationale.
Prefer exact rewrites; if one isn't possible, measure the cost and say so
rather than loosening `--atol` silently.
