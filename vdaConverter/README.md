# Converter for On-Device Video Depth Anything

Converts [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
(VDA) PyTorch checkpoints into TFLite models for on-device streaming
inference, with tooling to verify the export is numerically correct and to
benchmark it on a real Android device's CPU and GPU delegate.

This directory vendors the upstream VDA source under `Video-Depth-Anything/`
and adds a conversion/verification/benchmarking toolkit around it, plus a
second copy of the model source (`Video-Depth-Anything/video_depth_anything_gpu/`)
rewritten to be **fully compatible with the TFLite GPU delegate**. A
`--source gpu` export of `vits` at 720x1280 runs *entirely* on the GPU —
zero CPU fallback, no unsupported ops — at roughly **1.4 s/frame** on a
Samsung SM-S711B. See [The `original` vs. `gpu` split](#the-original-vs-gpu-split)
for what that took.

The layout takes direct inspiration from the sibling
[On-Device-RVM converter](../../../On-Device-RVM/converter/rvmConverter).

## Contents

- [Setup](#setup)
- [File structure](#file-structure)
- [Core concepts](#core-concepts)
- [Utilities](#utilities)
  - [`convert.py` — export to TFLite](#convertpy--export-to-tflite)
  - [`verify.py` — check a `.tflite` pair against PyTorch](#verifypy--check-a-tflite-pair-against-pytorch)
  - [`compare.py` — check `gpu` against `original`](#comparepy--check-gpu-against-original)
  - [`benchmark/` — on-device CPU/GPU benchmarking](#benchmark--on-device-cpugpu-benchmarking)
  - [`scripts/` — user-friendly wrappers](#scripts--user-friendly-wrappers)
- [The `original` vs. `gpu` split](#the-original-vs-gpu-split)
- [Typical workflow](#typical-workflow)
- [Claude Code skills](#claude-code-skills)
- [Gotchas](#gotchas)

## Setup

### 1. Conda environment

`environment.yaml` is a portable export of the exact environment this
toolkit was built and tested against (Python 3.11, `torch==2.12.0`,
`litert-torch==0.9.3`, `tensorflow`, etc.):

```bash
conda env create -n vda-convert -f environment.yaml
conda activate vda-convert
```

`requirements.txt` documents the pip equivalent, for those who'd rather not
use conda:

```bash
pip install -r requirements.txt
```

Every script under `scripts/` activates the `vda-convert` env for you (via
`conda activate vda-convert`), so once it exists under that exact name you
don't need to activate it by hand.

### 2. Checkpoints

`Video-Depth-Anything/checkpoints/` is gitignored and empty on a fresh
clone. Download the official VDA checkpoints per
`Video-Depth-Anything/get_weights.sh`, named `video_depth_anything_<variant>.pth`
(`vits`, `vitb`, `vitl`) to match `convert.py`'s default checkpoint lookup.

### 3. Android device (optional, for benchmarking only)

`benchmark/binary/` ships prebuilt TFLite `benchmark_model` binaries for two
ABIs (`android_aarch64_benchmark_model` for `arm64-v8a`,
`android_arm_benchmark_model` for `armeabi-v7a`/`armeabi`) — built per the
official LiteRT guide on [implementing/building a delegate and its benchmark
tooling](https://developers.google.com/edge/litert/performance/implementing_delegate).
`benchmark_cpu.sh`/`benchmark_gpu.sh` detect the connected device's ABI and
push the matching one automatically. You'll need `adb` on `PATH` and at
least one Android device reachable:

```bash
adb devices -l
```

A device with neither ABI (rare — some x86 emulators) can't run either
binary; the scripts detect this via `adb shell getprop ro.product.cpu.abi`
and abort with an explicit "Unsupported device architecture" error rather
than failing unhelpfully at push/exec time.

## File structure

```
vdaConverter/
├── Video-Depth-Anything/          # vendored upstream VDA source (never edited by this toolkit)
│   ├── video_depth_anything/      #   the streaming model source convert.py's "original" source traces
│   └── checkpoints/               #   gitignored; put video_depth_anything_<variant>.pth here
├── wrapper.py                     # InitWrapper/StepWrapper: the nn.Modules actually traced/exported
├── convert.py                     # PyTorch checkpoint -> .tflite (init + step signatures)
├── verify.py                      # exported .tflite pair vs. PyTorch, numerical check
├── compare.py                     # original vs. gpu, numerical check (pure PyTorch; gpu not implemented yet)
├── scripts/                       # user-friendly wrappers (env activation + cwd handled)
│   ├── convert.sh
│   ├── verify.sh
│   └── benchmark.sh
├── benchmark/                     # on-device CPU/GPU benchmarking via adb
│   ├── benchmark_cpu.sh
│   ├── benchmark_gpu.sh
│   ├── binary/                    # prebuilt per-ABI TFLite benchmark tools (see Setup step 3)
│   └── <original|gpu>/<cpu|gpu>/*.log   # generated logs, gitignored
├── tflite_models/                 # convert.py's output, gitignored
│   └── <original|gpu>/*.tflite
├── analysis/                      # generated op-graph visualizations, gitignored
├── environment.yaml               # conda env export
├── requirements.txt               # pip equivalent
└── .gitignore
```

## Core concepts

**`InitWrapper`/`StepWrapper` (`wrapper.py`)** are what actually get traced
and exported — not `VideoDepthAnything` directly. VDA is a *streaming*
model: it keeps a rolling cache of hidden state across frames, so exporting
it to TFLite's static-shape world needs two separate signatures instead of
one:

- **`InitWrapper`**: the first frame. No cache exists yet
  (`cached_hidden_state_list=None`), so it also seeds the cache by tiling
  the single-frame hidden state out to the full context length.
- **`StepWrapper`**: every later frame. Takes the previous fixed-shape cache
  in (`h0`–`h7`), returns the depth map and the updated cache.

Both wrappers:
- **Input** `frame`: `[B, H, W, 3]`, float, **0–255** range (NHWC). Resized
  internally (bicubic) to the model's native working resolution if
  `height`/`width` don't already match it, then ImageNet-normalized.
- **Output** `depth`: `[B, H, W, 1]`, resized back (bilinear) to the input
  resolution.
- **Cache slots** `h0`–`h7`: 8 fixed-shape tensors (4 motion modules × 2
  attention blocks each), `[N, context_len, C]`. `StepWrapper` rolls each
  one forward — drop the oldest slice, append the newest — every call.

**`variant`**: `vits`, `vitb`, or `vitl` — VDA's three encoder options.

**`source`**: `original` or `gpu` — which copy of the VDA model source to
trace from. See [below](#the-original-vs-gpu-split). Also picks the default
output directory, `tflite_models/<source>/`.

**Output naming convention** (`convert.py`'s `default_output()`), relied on
by `verify.py`'s filename inference and `benchmark/*.sh`'s log-folder
mirroring:

```
tflite_models/<source>/video_depth_<variant>_<height>x<width>_<init|step>.tflite
```

## Utilities

### `convert.py` — export to TFLite

```bash
python convert.py [options]        # or: ./scripts/convert.sh [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--variant` | `all` | `vits`, `vitb`, `vitl`, or `all` (converts all three) |
| `--source` | `original` | `original` or `gpu` (see [below](#the-original-vs-gpu-split)) |
| `--height` / `--width` | `720` / `1280` | traced (static) input resolution |
| `--input-size` | `518` | base working resolution before the aspect-ratio-preserving adjustment -- same role/default as upstream `infer_video_depth_one()`'s own `input_size` parameter |
| `--infer-len` | `8` | context length (frames) baked into the cache tensors -- overrides the source's own (much larger) native `INFER_LEN`, purely a wrapper-level choice |
| `--checkpoint` | derived | only valid with a single `--variant`, not `all` |
| `--output-dir` | `tflite_models/<source>` | where output filenames go |
| `--skip-verify` | off | skip the built-in PyTorch-vs-exported-model numerical check |

Run `python convert.py --help` for the full, current, authoritative list.

For each variant it traces `InitWrapper`/`StepWrapper` with
`litert_torch.signature(...).convert()` (signatures `"init"` and `"step"`),
prints the resolved cache shapes, runs a quick in-memory
PyTorch-vs-edge-model sanity check (unless `--skip-verify`), and writes both
`.tflite` files.

### `verify.py` — check a `.tflite` pair against PyTorch

```bash
python verify.py --tflite <path/to/model_init.tflite> [options]   # or: ./scripts/verify.sh ...
```

Pass either the `_init` or `_step` file — the sibling is found
automatically (same directory, `_init`/`_step` swapped). Builds the matching
PyTorch wrapper, runs it and the exported `.tflite` on the same synthetic
input, and prints a per-output PASS/FAIL table (`depth`, `h0`–`h7`) for
**both** signatures: `init` on a random frame, then `step` fed that same
call's own cache output. Scope is deliberately narrow — **wrapper vs
exported `.tflite`, i.e. conversion fidelity only**, matching the RVM
converter's `verify.py`. Comparing against the untouched original model is
`compare.py`'s job.

`.tflite` outputs are read through the model's named signature runner
(`ai_edge_litert`), not by sorting raw tensor `index` values — a tensor's
buffer index reflects internal flatbuffer layout, not `forward()`'s
return-value order, so index-sorting can silently pair the wrong tensors.

`--variant`, `--source`, `--height`, `--width`, `--input-size` and
`--infer-len` are all optional — they're inferred from `--tflite`'s
path/filename via the naming convention above, and it prints which flags it
inferred vs. which you passed. `--input-size`/`--infer-len` must match what
the pair was actually converted with; a mismatch silently compares apples to
oranges (large diffs, not a real regression).

`--atol` (default `1e-2`) sets the pass/fail threshold. Conversion fidelity
typically lands around `1e-5`.

### `compare.py` — check `gpu` against `original`

```bash
python compare.py --variant {vits,vitb,vitl} [options]
```

Pure PyTorch, no TFLite involved. **This is the tool to run while editing
`video_depth_anything_gpu/`** — it is what proves a GPU-compatibility
rewrite didn't change the model's behaviour.

It loads the **same checkpoint** three ways and runs all three on an
identical two-frame sequence, exercising `init` and `step`:

1. the untouched original model's own `infer_video_depth_one()`
2. `wrapper(source="original")`
3. `wrapper(source="gpu")`

The extra third leg (vs. RVM's two) means a bug introduced by the *wrapper*
shows up separately from one introduced by a `_gpu` edit.

Two tolerances, because the pairs have genuinely different expected floors:

| flag | default | applies to |
|---|---|---|
| `--atol` | `5e-5` | wrapper(original) vs wrapper(gpu) |
| `--code-atol` | `1e-3` | original code vs either wrapper |

`--code-atol` is never `0`: the wrapper resizes/normalizes in torch tensor
ops so `forward()` stays traceable, while upstream does the equivalent work
via cv2/numpy — two independent float implementations of the same formula.

`--atol` is `5e-5` rather than `0` for exactly one reason: the
`linear_as_conv1x1` substitution (see below). `nn.Linear` and a 1×1
`F.conv2d` are **bitwise identical up to a reduction length of 384** but
dispatch to differently-blocked CPU kernels beyond it, so `Mlp.fc2`
(`c_in=1536`) drifts ~`2.5e-6` per layer, compounding over DINOv2's 12
residual blocks to ~`3e-5`. That's floating-point reassociation, not a logic
difference — ~`6e-6` relative against typical depth values. **Every other
`_gpu` edit is exact and reports `0.000000`.**

### `benchmark/` — on-device CPU/GPU benchmarking

```bash
./benchmark/benchmark_cpu.sh <model.tflite> [device_name_or_id]
./benchmark/benchmark_gpu.sh <model.tflite> [device_name_or_id]
```

Queries the device's ABI (`adb shell getprop ro.product.cpu.abi`) to pick the
matching binary from `benchmark/binary/` (see [Setup step
3](#3-android-device-optional-for-benchmarking-only)), then pushes it and
the given `.tflite` to `/data/local/tmp/vda_benchmark/` on the device via
`adb` (`-s <device>` only if a device is given) and runs it —
`benchmark_cpu.sh` with `--num_threads=10`, `benchmark_gpu.sh` with
`--use_gpu=true` — both with `--enable_op_profiling=true --verbose=true`,
merging stdout **and** stderr into the saved log (stderr is where the
important `ERROR:` diagnostics live).

Logs mirror the model's location under `tflite_models/`: a model at
`tflite_models/<source>/foo.tflite` produces
`benchmark/<source>/<cpu|gpu>/foo_<device>.log` (outer layer = source, inner
layer = backend). A model outside `tflite_models/` falls back to a flat
`benchmark/<cpu|gpu>/foo_<device>.log`. Since only `init`/`step` pairs of
`.tflite` files exist, benchmark each half separately.

### `scripts/` — user-friendly wrappers

Thin wrappers that activate the `vda-convert` conda env and `cd` to the
project root (so relative paths work regardless of your current directory),
then forward everything else through:

```bash
./scripts/convert.sh [convert.py options]
./scripts/verify.sh --tflite <path> [verify.py options]
./scripts/benchmark.sh <cpu|gpu> <model.tflite> [device_name_or_id]
```

`convert.sh`/`verify.sh` pass `--help` straight through to the underlying
Python CLI. `benchmark.sh` has its own `-h`/`--help` (it has one more
required argument — the backend — that the other two don't) and dispatches
to `benchmark/benchmark_cpu.sh` or `benchmark_gpu.sh`.

## The `original` vs. `gpu` split

`Video-Depth-Anything/video_depth_anything/` is the **unmodified** upstream
VDA source — it must stay byte-identical to upstream and should never be
edited. `Video-Depth-Anything/video_depth_anything_gpu/` is a parallel copy
meant to be rewritten for TFLite GPU-delegate compatibility, the same role
`RobustVideoMatting/model_gpu` plays in the RVM converter -- registered in
`wrapper.py`'s `SOURCE_PACKAGES`:

```python
SOURCE_PACKAGES = {
    "original": "video_depth_anything",
    "gpu": "video_depth_anything_gpu",
}
```

`wrapper.py`'s `load_source()` resolves whichever package name is registered
via `importlib.import_module`, so `convert.py`, `verify.py`, and `compare.py`
all pick up any source registered here automatically -- no other code
changes needed to add another one.

### Status: fully delegated

A `--source gpu` export now reports:

```
INFO: Explicitly applied GPU delegate, and the model graph
      will be completely executed by the delegate.
```

No unsupported-op list, no CPU fallback, one delegate kernel. Measured on a
Samsung SM-S711B at 720x1280, `vits`, `--infer-len 8`:

| | init | step |
|---|---|---|
| latency | **1.375 s** | **1.515 s** |
| run-to-run std | 1.3 ms | 6.2 ms |

For contrast, the first build that merely *initialized* the delegate ran at
6.6 s with ~8% of nodes on GPU and ~175 ms of variance. The near-zero
variance is itself the signature of an uninterrupted GPU pipeline.

### The fixes

Every entry is an exact rewrite verified with `compare.py` against the
unmodified original, except where noted. All shape claims were confirmed by
reading the exported flatbuffer, **not** by reasoning about eager PyTorch
shapes — see [Gotchas](#gotchas) for why that distinction matters.

| File | Problem | Fix |
|---|---|---|
| `dinov2.py` | `CONCATENATION: Expected a 4D tensor of shape BxHxWxC but got 1x1x384` — **fatal**, killed delegate init outright | route the `cls_token` concat through an explicit 4D shape (unsqueeze → cat → squeeze) |
| `dinov2.py` | positional-encoding `F.interpolate` re-run every forward | `precompute_pos_encoding()` caches it at build time — it only depends on fixed `(w, h, dtype)` and the frozen `pos_embed` |
| `dinov2_layers/attention.py` | rank-5 qkv pack ⇒ `SLICE` op version 5 (delegate max 2), plus rank-5 `TRANSPOSE`/`RESHAPE` | split the projection **before** reshaping to heads, keeping every intermediate ≤4D. Drops `SLICE` v5→v1 and `TRANSPOSE` v4→v1 |
| `dinov2_layers/attention.py`, `mlp.py`, `gpu_compat.py` | `MUL: Doesn't support broadcasting - input0: [2443,384], input1: [1,1,384]` — LayerScale, fencing off the whole backbone | apply `proj`/`fc2` as 1×1 convs (`linear_as_conv1x1`). **The single biggest win: 102 → 836 delegated nodes, 6.6 s → 2.0 s.** Not bit-exact — see `compare.py` above |
| `motion_module/attention.py` | `DIV: No support of few identical inputs` — softmax over a length-1 sequence lowers to `y/y` | short-circuit the degenerate case: softmax of one logit is exactly 1.0, so `probs @ value == value`. Also removed ~124 nodes |
| `motion_module/motion_module.py`, `attention.py` | `ADD: Doesn't support broadcasting - input0: [2442,192], input1: [1,2442,192]` — residual adds | apply `to_out[0]` and `FeedForward`'s final projection as 1×1 convs |
| `motion_module/motion_module.py` | `GATHER_ND` from `nn.GroupNorm` — the last unsupported op | the gather was an **identity gather**: affine params indexed by a constant `arange([C])` purely to reshape `[C]`→`[1,C]`. Normalize without affine params, then apply scale/shift with an explicit reshape |

Two converter-level changes live outside the model tree, in `wrapper.py`:

- **Cache as 4D `[1, N, T, C]`** (see [Core concepts](#core-concepts)) — required before the `ADD` fix could land without tripping `Batch size mismatch`.
- **`.expand()` → `torch.cat`** for cache seeding — the delegate has no `BROADCAST_TO` kernel at all, and `.expand()` lowers to exactly that even followed by `.contiguous()`.

### The resize, and why arbitrary input shapes work

TFLite has no bicubic op, so `F.interpolate(mode="bicubic")` lowers to a
data-dependent `GATHER_ND` plus `BROADCAST_TO`/`CONCATENATION` to build the
sampling grid — none supported. That used to force any non-native
`--height`/`--width` to fall back to CPU.

But **the scale factor is fixed at conversion time**, which makes bicubic a
*constant linear operator*, and PyTorch's implementation is separable — so
it factors exactly into two fixed matrices. `wrapper.py`'s
`_resize_axis_matrix()` recovers them by pushing basis vectors through
`F.interpolate` itself, so PyTorch's exact kernel weights and boundary
handling come along for free rather than being reimplemented.

Those are applied as two **1×1 convolutions**, rotating the resampled axis
into the channel position. (A plain matmul was tried first and rejected with
`BATCH_MATMUL: Not supported batched mat mul case` — the delegate won't
broadcast a 2D matrix against a 4D activation. `CONV_2D` has no such limit.)

The practical consequence: **you can convert at any input resolution with
full GPU compatibility and no accuracy loss.** Agreement with
`F.interpolate` is ~`3.6e-07`; cost is 6.2 MB of constants and ~3.3 GMAC per
frame, which did not show above the noise floor on device. Switching to
bilinear instead — the obvious cheap alternative — was measured at **0.35
max depth error** on values around 4.5, i.e. ~800,000× worse, and was
rejected.

This applies to `--source gpu` only; `original` keeps plain `F.interpolate`.

## Typical workflow

```bash
# 1. Convert (all three variants, original source)
./scripts/convert.sh

# 2. Verify each export is numerically correct
./scripts/verify.sh --tflite tflite_models/original/video_depth_vits_720x1280_init.tflite
./scripts/verify.sh --tflite tflite_models/original/video_depth_vitb_720x1280_init.tflite

# 3. Benchmark on-device
./scripts/benchmark.sh cpu tflite_models/original/video_depth_vits_720x1280_init.tflite <device>
./scripts/benchmark.sh cpu tflite_models/original/video_depth_vits_720x1280_step.tflite <device>
```

## Claude Code skills

`.claude/skills/` has project-scoped skills (auto-discovered by Claude Code
from this repo, no extra setup) documenting these workflows in more
operational detail than this README:

- `vda-setup` — this repo's first-time setup
- `vda-convert` — `convert.py`/`scripts/convert.sh` in depth
- `vda-verify` — `verify.py`/`scripts/verify.sh` in depth (wrapper vs `.tflite`)
- `vda-compare` — `compare.py` in depth (original vs `original`/`gpu` wrappers);
  the tool to run while editing `video_depth_anything_gpu/`
- `vda-benchmark` — on-device benchmarking in depth
- `vda-gpu-delegate-fix` — the recipe for diagnosing and fixing a
  GPU-delegate-unsupported op, with every fix from the port as a worked
  example and the pitfalls that cost the most time

## Gotchas

- Never edit `Video-Depth-Anything/video_depth_anything/*.py` — it's
  read-only vendored upstream source. `video_depth_anything_gpu/` is where
  GPU-delegate-compatibility edits belong instead (see
  [above](#the-original-vs-gpu-split)).

- **Eager PyTorch shapes are not evidence about the exported graph.** This
  is the single most expensive lesson here. `torch.export` + litert's
  lowering re-ranks tensors aggressively: `nn.Linear` becomes a
  `FULLY_CONNECTED` whose output is canonicalized to 2D `[N, C]`, and a
  size-1 leading axis gets squeezed away. A `LayerScale` fix written as
  `gamma.view((1,)*(x.dim()-1) + (dim,))` looked obviously right and was a
  **complete no-op** — in eager `x.dim()` was 3, so it re-created the exact
  `[1,1,384]` shape that was already failing, while the real mismatch came
  from the exporter flattening the *other* operand. Always confirm a shape
  fix by reading the flatbuffer.

- **Small repro scripts mislead on this particular problem.** Two separate
  probes gave confidently wrong answers: one produced *zero* `MUL` ops
  because constant-folding fused the scale into the Linear weights (which
  doesn't happen in the real block), and another correctly reproduced the
  baseline mismatch *and* the exporter's squeeze behaviour, yet still
  mispredicted the winning fix because surrounding ops change how the graph
  canonicalizes. Full conversion is the only reliable oracle.

- **The delegate reads a tensor's first dimension as batch size.** Any fix
  that makes an op same-shape by flattening to 2D `[N, C]` declares
  `batch=N`, which collides with the batch-1 tensors around it and kills
  delegate init entirely (`Batch size mismatch, expected 627 but got 1`).
  That is strictly *worse* than the original mismatch: the whole graph falls
  back to CPU rather than just the offending nodes. Rank-match **upward**,
  keeping batch 1.

- **Grep the benchmark log for more than `ERROR`.** Unsupported ops are
  reported in two different wordings — `Operation is not supported` and
  `OP is supported, but tensor type/shape isn't compatible` — and the second
  contains neither "ERROR" nor "not supported". Matching only the first
  pattern will make you think ops disappeared when they didn't.

- **A fix that clears an op category can still be a regression.** Removing
  one blocker often just exposes a later one that was previously hidden
  behind it (this happened twice: `DIV` and `Batch size mismatch` both
  surfaced only after the `ADD` fix grew the partition). Always re-run the
  device benchmark, not just the flatbuffer check.
- `benchmark/binary/` holds one prebuilt `benchmark_model` per ABI
  (`arm64-v8a`, `armeabi-v7a`/`armeabi`) — see [Setup step
  3](#3-android-device-optional-for-benchmarking-only) for how they were
  built. `benchmark_cpu.sh`/`benchmark_gpu.sh` auto-detect the device's ABI
  and push the right one; a device on neither ABI gets an explicit
  "Unsupported device architecture" error instead of a confusing push/exec
  failure.
- A `.tflite` "model" here is always a **pair** of files (`_init`/`_step`)
  — `verify.py`/`benchmark/*.sh` both need the actual on-device streaming
  loop (init once, then step per subsequent frame) to be exercised, not
  just one signature in isolation.
- `benchmark/`, `tflite_models/`, and `analysis/` are all gitignored
  (generated outputs); `benchmark/binary/benchmark_model` and the `.sh`
  scripts themselves are tracked.
