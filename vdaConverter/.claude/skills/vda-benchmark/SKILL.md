---
name: vda-benchmark
description: Benchmark an exported VDA .tflite model on a real Android device's CPU or GPU delegate, via benchmark/*.sh / scripts/benchmark.sh. Use when asked to measure on-device inference speed, check GPU-delegate compatibility/op-support, or profile a converted model.
---

# Benchmarking a VDA `.tflite` model on-device

```bash
./scripts/benchmark.sh <cpu|gpu> <model.tflite> [device_name_or_id]
# or directly:
./benchmark/benchmark_cpu.sh <model.tflite> [device_name_or_id]
./benchmark/benchmark_gpu.sh <model.tflite> [device_name_or_id]
```

`device_name_or_id` is optional if exactly one device is connected via
`adb`. Since a converted model is always an `init`/`step` **pair** (see
`vda-convert`), benchmark both files separately -- they're different
graphs with different input/output shapes.

Under the hood: detects the device's ABI (`adb shell getprop
ro.product.cpu.abi`), picks the matching prebuilt binary from
`benchmark/binary/`, pushes it and the model to
`/data/local/tmp/vda_benchmark/`, and runs it with
`--enable_op_profiling=true --verbose=true` (`benchmark_cpu.sh` adds
`--num_threads=10`; `benchmark_gpu.sh` adds `--use_gpu=true`). Both stdout
and stderr are merged into the saved log -- stderr is where the important
`ERROR:` diagnostics live.

## Where logs go

Mirrors the model's location under `tflite_models/`: a model at
`tflite_models/<source>/foo.tflite` produces
`benchmark/<source>/<cpu|gpu>/foo_<device>.log` (outer layer = source,
inner layer = backend). A model outside `tflite_models/` falls back to a
flat `benchmark/<cpu|gpu>/foo_<device>.log`.

## Reading the output

- `INFO: Explicitly applied GPU delegate, and the model graph will be
  completely executed by the delegate.` -- full delegation, zero CPU
  fallback. **This is the current expected state for a `--source gpu`
  build**; if it's missing, something regressed.
- `ERROR: Following operations are not supported by GPU delegate: ...` -- a
  *soft* failure; those specific ops fall back to CPU, the rest still runs
  on GPU. See the `vda-gpu-delegate-fix` skill.
- `TfLiteGpuDelegate Init: ... / ERROR: Benchmarking failed.` -- a *hard*
  failure; the delegate couldn't prepare the graph at all, so **nothing**
  runs on GPU. Strictly worse than the soft case.
- `N operations will run on the GPU, and the remaining M ... on the CPU` --
  the delegation ratio, the number to watch when chasing coverage.
- `Timings (microseconds): count=N ... avg=... std=...` -- the headline
  inference time. Watch `std` too: a fully-delegated graph shows single-digit
  milliseconds of variance, while heavy CPU fallback shows hundreds.

**Grep for more than `ERROR`.** Unsupported ops appear in two wordings, and
the second contains neither "ERROR" nor "not supported":

- `<OP>: Operation is not supported.`
- `<OP>: OP is supported, but tensor type/shape isn't compatible.`

Matching only the first will make you think ops disappeared when they
didn't. Capture the whole block instead:

```bash
sed -n '/not supported by GPU delegate/,/operations will run/p' <log>
```

## Reference numbers

`vits`, 720x1280, `--infer-len 8`, `--source gpu`, on a Samsung SM-S711B:
**init ~1.375 s, step ~1.515 s**, fully delegated, `std` of 1-6 ms. An
`--source original` build of the same model is *not* GPU-friendly and will
show a long unsupported-op list -- that's expected, not a regression.

If a `gpu`-source build reports unsupported ops or fails delegate init, see
the `vda-gpu-delegate-fix` skill for the diagnosis recipe.
