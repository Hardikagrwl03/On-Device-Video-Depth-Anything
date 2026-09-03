---
name: vda-convert
description: Convert a Video Depth Anything PyTorch checkpoint (vits/vitb/vitl) to a pair of TFLite models (init + step signatures) via convert.py / scripts/convert.sh. Use when asked to export, convert, or trace a VDA model to TFLite, or to change its traced resolution.
---

# Converting VDA checkpoints to TFLite

`convert.py` traces `wrapper.InitWrapper` and `wrapper.StepWrapper` (which
together wrap `VideoDepthAnything`'s streaming interface) with
`litert_torch.signature(...).convert()` and exports two `.tflite` files per
variant -- VDA is a streaming model, so a static-shape export needs a
first-frame signature (`init`, no cache yet) and a later-frame signature
(`step`, fixed-shape cache in and out) rather than one. Prefer running it
via the wrapper script, which activates the `vda-convert` conda env and
`cd`s to the repo root for you (see `vda-setup` if the env doesn't exist
yet):

```bash
./scripts/convert.sh [options]
```

(equivalent to `conda activate vda-convert && python convert.py [options]`
from the repo root.)

## Key options

| Flag | Default | Meaning |
|---|---|---|
| `--variant` | `all` | `vits`, `vitb`, `vitl`, or `all` (converts all three) |
| `--source` | `original` | `original` traces `Video-Depth-Anything/video_depth_anything` (unmodified upstream); `gpu` traces `Video-Depth-Anything/video_depth_anything_gpu`, rewritten for TFLite GPU-delegate compatibility. **Use `gpu` for anything that will run on-device** -- it is fully delegated, whereas an `original` build falls back heavily to CPU. See `vda-gpu-delegate-fix` and README.md's "original vs gpu split" |
| `--height` / `--width` | `720` / `1280` | traced (static) input/output resolution |
| `--input-size` | `518` | base working resolution before the aspect-ratio-preserving adjustment in `compute_target_size()` -- same role/default as upstream `infer_video_depth_one()`'s own `input_size` parameter; see below |
| `--infer-len` | `8` | context length (frames) baked into the cache tensors -- overrides the source's own (much larger) native `INFER_LEN`, purely a wrapper-level choice; see below |
| `--checkpoint` | derived | only valid with a single `--variant` (not `all`); otherwise derived automatically (see naming below) |
| `--output-dir` | `tflite_models/<source>` | where derived output filenames go |
| `--skip-verify` | off | skip the built-in PyTorch-vs-exported-edge-model numerical check that runs right after tracing |

Run `./scripts/convert.sh --help` for the exact, current flag list --
`convert.py`'s CLI is the source of truth, this table can drift.

## Output naming convention

Files are named and located by convention -- other tools in this repo
(`verify.py`'s filename inference, `benchmark/*.sh`'s folder mirroring) rely
on this:

```
tflite_models/<source>/vda_<variant>_<height>x<width>_input<input_size>_infer<infer_len>_<init|step>.tflite
```

E.g. `tflite_models/gpu/vda_vits_720x1280_input518_infer8_init.tflite` and
its sibling `..._step.tflite`.

## What the wrappers do

`InitWrapper`/`StepWrapper` (in `wrapper.py`) take `frame` as `[B, H, W, 3]`,
`0-255`-ranged float (NHWC) and return `depth` as `[B, H, W, 1]`, at the
same `height`/`width` the input came in at (resized internally if the
model's native working resolution differs). The 8 cache slots (`h0`-`h7`,
4 motion modules x 2 attention blocks) are `[N, context_len, C]` tensors --
`InitWrapper` seeds them by tiling the first frame's hidden state,
`StepWrapper` rolls each one forward (drop oldest, append newest) every
call. `context_len` is `--infer-len - 1` (default `8 - 1 = 7`) -- a
deliberate wrapper-level choice, *not* derived from the source's own native
`INFER_LEN` (32 in the unmodified upstream model, which is far larger than a
fixed-shape TFLite export can practically afford). `target_h`/`target_w`
(the actual traced working resolution) are `compute_target_size(height,
width, Resize, input_size)` -- `--input-size` is the same knob as upstream
`infer_video_depth_one()`'s own `input_size` argument, just exposed here
instead of hardcoded to `518`. Both `target_h`/`target_w` *and* `input_size`
itself are stored on the wrapper instance (`self.target_h`, `self.input_size`,
etc.) even though only `target_h`/`target_w` affect `forward()` -- `input_size`
is kept purely so the wrapper is self-describing about which resolution it
was built for.

Because `--input-size` and `--infer-len` determine the traced shapes, they
are baked into the output *filename* -- which is how `verify.py` recovers
them automatically. You only need to pass them by hand if you also override
`--output-dir` with a path that breaks the naming convention; a mismatch
there silently compares two different configurations and produces huge diffs
that look like a regression but aren't.

## After converting

1. `vda-verify` -- confirm the `.tflite` matches the PyTorch wrapper it was
   traced from (conversion fidelity).
2. `vda-compare` -- if you edited `video_depth_anything_gpu/`, confirm the
   edit didn't change the model's behaviour vs. the untouched original.
3. `vda-benchmark` -- measure on device, benchmarking `init` and `step`
   separately.
4. `vda-visualize` -- render an op-graph image, `init` and `step` separately.

`./run.sh` chains all four of the above (plus this conversion step) for one
variant/source/resolution combination in a single command -- defaults to
`--variant vits --source gpu --backend gpu`. Reach for it instead of the
individual scripts when you just want the full pipeline run once, end to
end; use the individual scripts (and skills) directly when iterating on one
step.

For a `--source gpu` build that reports unsupported ops or fails delegate
init, see `vda-gpu-delegate-fix`.
