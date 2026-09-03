---
name: vda-verify
description: Verify an exported VDA .tflite model pair (init + step) against the PyTorch wrapper it was traced from, via verify.py / scripts/verify.sh. Use when asked to check that a conversion is numerically faithful, or to debug a suspicious diff between a .tflite file and PyTorch. For comparing the gpu model source against the untouched original, use vda-compare instead.
---

# Verifying an exported VDA `.tflite` pair

```bash
python verify.py --tflite <path/to/model_init.tflite> [options]   # or: ./scripts/verify.sh ...
```

Pass either the `_init` or `_step` file from a pair -- the sibling is found
automatically in the same directory (`_init`/`_step` swapped; see
`vda-convert`'s output naming convention).

## What it checks (and what it doesn't)

**Scope is deliberately narrow: PyTorch wrapper vs exported `.tflite`, i.e.
conversion fidelity only.** It builds the matching wrapper from the
parameters encoded in the filename, runs it and the `.tflite` on the same
synthetic input, and prints a per-output PASS/FAIL table (`depth`, `h0`-`h7`)
for **both** signatures:

1. `init` on a random `[1, H, W, 3]` frame,
2. `step` fed that same call's own cache output.

Feeding step the init call's real cache means the step comparison exercises
a genuine non-degenerate cache, while both sides still receive identical
inputs -- so a diff isolates the conversion, not the cache seeding.

It does **not** compare against the untouched original model. That is
`compare.py`'s job -- see the `vda-compare` skill. (An older version of this
script did all three at once; it was split so each tool has one clear
reference point.)

Outputs are read through the model's named signature runner
(`ai_edge_litert.interpreter.Interpreter`), never by sorting raw tensor
`index` values -- a tensor's buffer index reflects internal flatbuffer
layout, not `forward()`'s return-value order, so index-sorting can silently
pair the wrong tensors against each other.

## Key options

| Flag | Default | Meaning |
|---|---|---|
| `--variant` / `--source` / `--height` / `--width` / `--input-size` / `--infer-len` | inferred | all pulled from the `--tflite` filename/path when not given -- see `vda-convert`'s naming convention. The script prints which it inferred vs. which you passed |
| `--checkpoint` | derived | `Video-Depth-Anything/checkpoints/video_depth_anything_<variant>.pth` |
| `--atol` | `1e-2` | max allowed per-tensor absolute difference before a FAIL |

Run `./scripts/verify.sh --help` for the exact, current flag list.

Typical healthy output is `max_diff` around `1e-5` on every tensor.

## Debugging a FAIL

- **A large diff here means the export itself changed the model's
  behaviour**, not an architectural approximation -- there is no
  approximation left in this comparison, both sides are the same wrapper.
  Look for ops without exact TFLite-CPU equivalents, or a shape/dtype
  mismatch introduced by the wrapper.

- **Check `--input-size` / `--infer-len` first.** They are normally inferred
  from the filename, but if you pass them explicitly they must match what
  the pair was actually converted with, since they determine the wrapper's
  `context_len` and `target_h`/`target_w`. A mismatch silently compares two
  different configurations and produces huge diffs that look like a
  regression but aren't.

- **`--source` is inferred from a *directory* component of the path**
  (`tflite_models/<source>/...`). If you converted with a custom
  `--output-dir` that doesn't contain `gpu` or `original` as a path segment,
  inference silently falls back to `original` and will build the wrong
  wrapper. Pass `--source` explicitly in that case.

`./run.sh` runs this (plus convert/compare/benchmark/visualize) for one
variant/source in a single command, always passing `--source` explicitly
(sidestepping the inference caveat above) -- see `vda-convert`.
