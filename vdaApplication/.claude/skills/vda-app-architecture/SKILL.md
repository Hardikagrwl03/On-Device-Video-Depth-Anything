---
name: vda-app-architecture
description: How the VDA app is structured and the invariants you must not break when editing it -- the generic module pattern, the init/step two-graph lifecycle, per-object thread confinement, and several traps that look harmless but deadlock, crash, or silently produce wrong output. Use before modifying DepthModule, DepthHiddenStates, DepthConfig, TFLiteModelRunner, VideoFrameDecoder/Encoder, DepthColormap, Controller, or DepthViewModel.
---

# VDA app: architecture and invariants

Read this before editing anything under `depth/`, `modelRunner/`, `video/`,
`utils/DepthColormap.kt` or `Controller.kt`. Most of the rules below exist
because the obvious version was tried first and broke.

## The generic module pattern

New models plug in without touching surrounding code:

- `ModuleInterface<Config : ConfigInterface, IO>` -- `configure/run/reset/close`.
- Each module brings its own config data class (`DepthConfig`) and its own named
  buffer holder (`DepthIO`), so callers get typed fields, not positional buffers.
- `HiddenStatesInterface` -- recurrent state buffers.
- `ModelRunnerInterface` / `TFLiteModelRunner` -- the shared TFLite wrapper.
  Rebuilds the interpreter only when model file, compute device, or thread count
  actually changed.

`Controller` orchestrates: one `VideoFrameDecoder`, one `DepthModule`, and
**two** `VideoFrameEncoder`s (grayscale / colormap), so one decode+inference
pass writes two videos.

## VDA is two graphs, and `DepthModule` hides it

The exported model is an `init` graph (frame -> depth + seeded caches) and a
`step` graph (frame + caches -> depth + updated caches). `DepthModule` owns
**two** `TFLiteModelRunner`s and an `isSequenceStarted` flag:

- First `run` after `configure`/`reset` -> `init`. Then `depthInitModel.close()`:
  ~120 MB of weights nothing needs until the next reset.
- Every later `run` -> `step`, copying the fresh caches back into
  `hiddenStates` via a temporary `DepthHiddenStates` and `put()`.
- `reset()` zeroes the caches **and** clears `isSequenceStarted`. Zeroed caches
  are not a valid sequence start for VDA -- the next frame must go through
  `init` again. `TFLiteModelRunner.configure()` rebuilds a closed interpreter
  for free (it checks `interpreter == null`), which is how `init` comes back.

`DepthConfig` carries `initRuntimeConfig`, derived from `runtimeConfig` in
`init {}` so one device/thread choice covers both graphs. Keep it derived.

## Hidden states are token caches with derived shapes

`DepthHiddenStates` holds eight `[1, tokens, context, channels]` tensors. The
token count comes from the ViT patch grid at the model's **working**
resolution (`DepthConfig.targetHeight/targetWidth`, 518x924 for 720x1280 --
re-derived exactly as the converter's `compute_target_size()`), scaled per
DPT stage by `GRID_SCALES = [1,1,-2,-2,1,1,2,2]` (negative = divide, rounding
up). Channels come from `DepthConfig.Variant.channels`. For `vits` that gives
2442/2442/627/627/2442/2442/9768/9768 tokens, matching the step model's
signature exactly. If a new export changes any of this, `logSignature()`
output is the ground truth -- update the derivation, not a literal.

The eight states total ~80 MB; `zero()` clears them in 64 KB chunks, not via
a scratch array the size of the buffer.

## Thread confinement -- the core constraint

Every stateful native resource is pinned to one dedicated thread via
`utils/ConfinedRunner.kt`. The TFLite **GPU delegate binds an EGL context to
whichever thread built the interpreter**, so building and invoking must
happen on the same one. Threads are named `vda-depth`, `vda-decode`,
`vda-encode-gray`, `vda-encode-color`.

### Trap 1: re-entrancy deadlocks the confined thread

`ConfinedRunner` wraps a single-thread executor and **blocks** on the result. A
task already running on that thread must never call a public (wrapped) method
of the same object. Hence the `Impl` split:

```kotlin
override fun configure(c: DepthConfig) = runner.run { configureImpl(c) }
private fun configureImpl(c: DepthConfig) { resetImpl() /* NOT reset() */ }
```

**When adding a method: public wrapper -> `runner.run { ...Impl() }`, and every
internal call goes to the `Impl`.** Same rule in `VideoFrameEncoder`.

### Trap 2: `saveVideo()` must not shut the runner down

`Controller` reuses its encoders across runs; `close()` is called only from
`Controller.close()`. Likewise `Controller.depthVideo()` ends with
`depthModule.reset()`, never `close()` -- closing shuts the module's executor
for good and the next run throws `RejectedExecutionException`.

### Trap 3: never download or do unbounded I/O on a confined thread

`TFLiteModelRunner.loadModelFile` memory-maps from `ModelStore` -- fine.
Nothing in `depth/` or `modelRunner/` may trigger a network fetch.

## Correctness traps that produce *plausible-looking* wrong output

### Trap 4: the GPU delegate must stay FP32

`TFLiteModelRunner.newGpuDelegate()` builds
`GpuDelegate(GpuDelegateFactory.Options().setPrecisionLossAllowed(false))`.
The no-arg `GpuDelegate()` allows FP16, which produces NaN in VDA's motion
modules on real hardware. Do not "simplify" this back. See `vda-app-models`.

### Trap 5: the encoder is NV12 and reads R,G,B -- do not "fix" it back

`VideoFrameEncoder.floatBufferToNV12` writes chroma **U,V** (what
`COLOR_FormatYUV420SemiPlanar` means) and reads each pixel's floats as R,G,B.
An earlier version wrote V,U and read B,G,R; the two swaps cancel for RGB
input, which is why it went unnoticed, but they inverted the inferno ramp
(near came out cyan). `DepthColormap.renderColor` therefore writes plain RGB.

### Trap 6: input must match the model, and the check is deliberate

`Controller.depthVideo()` `require`s the decoded video's dimensions equal
`config.height x config.width`. Without it a mismatched clip overflows a
buffer deep inside the decoder with nothing to explain why. Keep the message
naming both sizes -- the user guide quotes it.

### Trap 7: `DepthColormap` normalises per frame

Raw output is unbounded relative inverse depth (larger = nearer); a constant
`range` is guarded (`>1e-6f`, else 1) so a flat frame renders black instead
of NaN. Per-frame min/max is a documented trade-off (brightness flicker across
cuts) chosen to keep the run single-pass; a per-clip normalisation needs a
second pass or buffered depth and is a design change, not a tweak.

## `DepthViewModel` traps

### Trap 8: `isConfiguring` must be raised synchronously

Set **before** `viewModelScope.launch`, cleared in `finally`. Otherwise
`updateConfig`'s guard reads a stale `false` and starts a second, overlapping
GPU-delegate build; the loser leaks its delegate.

### Trap 9: configuration failures must not throw

`runConfigure` catches and reports through `transientMessage`. An uncaught
throw propagates out of `viewModelScope` and kills the process.

## Kotlin/perf notes

- **`override` methods cannot declare default parameter values.** Interfaces
  declare the defaults; concrete overrides omit them.
- **Never use per-element `FloatBuffer.get(i)`/`put(i, v)` on a `SharedMemory`
  buffer.** Bulk-transfer into a `FloatArray`, loop over the primitive array,
  bulk-write back -- `DepthColormap` and `VideoFrameEncoder.floatBufferToNV12`
  both do this, reusing scratch arrays across frames.
- `DepthConfig` is a `data class` whose `init {}` rebuilds both file names and
  the working resolution, so `copy()` is safe -- there is no sentinel to
  preserve. But `buildModelFileName()` must stay byte-identical to
  `ModelSpec`'s name construction in `models/ModelManifest.kt`.
