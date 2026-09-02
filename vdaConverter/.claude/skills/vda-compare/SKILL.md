---
name: vda-compare
description: Compare the untouched original VDA model, wrapper(source='original') and wrapper(source='gpu') on identical inputs via compare.py -- pure PyTorch, no TFLite. Use when editing Video-Depth-Anything/video_depth_anything_gpu/ to confirm a GPU-delegate-compatibility rewrite did not change the model's behaviour, or when a depth output looks wrong and you need to tell a wrapper bug from a _gpu-source bug.
---

# Comparing the original model against the wrappers

```bash
python compare.py --variant {vits,vitb,vitl} [options]
```

**This is the tool to run while editing `video_depth_anything_gpu/`.** It is
what proves a GPU-compatibility rewrite is behaviour-preserving. Pure
PyTorch -- no TFLite, no device, so it is fast enough to run after every
edit.

## What it compares

Loads the **same checkpoint** three ways and runs all three on an identical
two-frame sequence, exercising `init` (frame 1) and `step` (frame 2, fed the
init call's real cache):

1. the untouched original model's own `infer_video_depth_one()`
2. `wrapper(source="original")`
3. `wrapper(source="gpu")`

The third leg is the point: it separates *"the wrapper is wrong"* from
*"a `_gpu` edit is wrong"*, which a two-way comparison cannot do. Because
`infer_video_depth_one()` doesn't expose its internal cache, the two
original-code comparisons cover `depth` only; the wrapper-vs-wrapper
comparison covers all 9 outputs (`depth`, `h0`-`h7`).

## The two tolerances

| Flag | Default | Applies to | Expected |
|---|---|---|---|
| `--atol` | `5e-5` | wrapper(original) vs wrapper(gpu) | `0.000000` for nearly every edit |
| `--code-atol` | `1e-3` | original code vs either wrapper | `~2e-5` |

**`--code-atol` is never `0`** and shouldn't be: the wrapper resizes and
normalizes in torch tensor ops so `forward()` stays traceable for export,
while upstream does the equivalent work via cv2/numpy. Two independent
floating-point implementations of the same formula are never bit-identical.

**`--atol` is `5e-5` rather than `0` for exactly one reason**: the
`linear_as_conv1x1` substitution (`dinov2_layers/gpu_compat.py`). `nn.Linear`
and a 1x1 `F.conv2d` are bitwise identical **up to a reduction length of
384** but dispatch to differently-blocked CPU kernels beyond it, so
`Mlp.fc2` (`c_in=1536`) drifts ~`2.5e-6` per layer, compounding over
DINOv2's 12 residual blocks to ~`3e-5`.

Every other `_gpu` edit is exact. **If a change you just made pushes
wrapper-vs-wrapper above `0.000000`, that is a signal, not noise** -- unless
you knowingly added another large-reduction `linear_as_conv1x1` site, treat
it as a bug in the rewrite rather than reaching for a looser `--atol`.

## Interpreting a FAIL

- **wrapper(original) vs wrapper(gpu) fails** -- the `_gpu` edit changed the
  math. This is the common case and the one this tool exists to catch.
- **Both original-code comparisons fail together, wrapper-vs-wrapper
  passes** -- the bug is in `wrapper.py` (shared by both sources), not in
  `video_depth_anything_gpu/`.
- **Only the `gpu` original-code comparison fails** -- shouldn't happen if
  wrapper-vs-wrapper passed; re-run before investigating.

## Where this sits in the loop

`compare.py` clean is a **precondition**, not a completion: it proves the
edit is behaviour-preserving, but says nothing about whether it achieved
anything for the GPU delegate. Confirm that separately by reading the
exported flatbuffer and benchmarking on device -- see `vda-gpu-delegate-fix`.
