---
name: vda-gpu-delegate-correctness
description: Diagnose a VDA model that is fully GPU-delegated (100% coverage, zero benchmark_gpu.sh errors, compare.py/verify.py both clean) but still produces wrong or NaN output on a real Android device's GPU. Use when depth or cache outputs are NaN/garbage on-device GPU but correct on-device CPU, or correct in every host-side tool -- none of which exercise the real device GPU delegate. For unsupported-op / CPU-fallback / delegate-init failures instead, use vda-gpu-delegate-fix.
---

# Diagnosing a silent GPU delegate correctness bug

**None of this repo's host-side tools can catch this bug class.**
`compare.py` is pure PyTorch (no TFLite at all). `verify.py` runs the
`.tflite` through `ai_edge_litert.interpreter.Interpreter`, which is
CPU/XNNPACK -- it has no GPU delegate path. `benchmark_gpu.sh`'s
delegation-coverage number only says every op *ran* on the delegate, not
that it computed the right answer. A model can pass all three perfectly
and still be silently wrong -- this happened for real: `nn.LayerNorm`'s
`MEAN`-axis reduction pattern was computed correctly in eager PyTorch and
via CPU/XNNPACK, and silently wrong (too-large variance, no error) only on
the actual GPU delegate kernel on real hardware.

**Only a real device, with the real GPU delegate, comparing real
intermediate tensors against CPU, can catch this.** There is no shortcut
through the host toolchain.

## The two known bug classes (check both, independently)

1. **A specific delegate kernel is buggy for a specific op/axis pattern.**
   Confirmed example: `nn.LayerNorm` on a `[1, N, C]` tensor decomposes to
   `MEAN` with `axis=[0, 2]` (batch axis + channel axis together, skipping
   the token axis) -- the GPU delegate's `MEAN` kernel returns the wrong
   variance for that specific non-trailing multi-axis pattern. Fix: force
   a numerically-equivalent but differently-shaped reduction (e.g. a
   single trailing-axis `dim=-1` reduction) and verify exactness with
   `compare.py`, same discipline as `vda-gpu-delegate-fix`.

2. **FP16 precision loss.** The GPU delegate's default
   `GpuDelegateFactory.Options()` allows FP16 compute
   (`setPrecisionLossAllowed(true)`). This is a **delegate configuration
   the consuming app controls** -- nothing in this converter's `.tflite`
   output can fix it. Test it in ~30 seconds with **zero reconversion**:
   run the same model on-device with `setPrecisionLossAllowed(false)`
   (Kotlin) or `--gpu_precision_loss_allowed=false` (`benchmark_model`).
   If that alone clears the NaN, it's this. If it doesn't, it's class 1
   above (or both).

**Test both, and test them independently -- do not assume one fix
subsumes the other.** They compound: reverting the class-1 LayerNorm fix
and forcing FP32 (class-2 fix) still produced wildly wrong *finite*
output on real hardware (a cache tensor with `min=-461,585,440` where CPU
gives `~-3`) -- not NaN, proof the two bugs are genuinely independent and
both fixes are required.

## The loop

Requires a real Android device and a consuming app project (not just this
converter repo) -- the GPU delegate only exists on-device.

1. **Write an instrumented debug wrapper.** For a bug already narrowed to
   roughly where it lives (see bisection below), monkeypatch an
   instrumented copy of the relevant `nn.Module.forward()` onto the real
   module instance via `types.MethodType`, adding `capture[name] = tensor`
   side effects at the points of interest -- a faithful copy of the real
   method body, not a rewrite, so the traced graph is unaffected. Wrap the
   whole model in a small `nn.Module` that clears the capture dict, calls
   the real forward, and returns `(*original_outputs, *captured_taps)`.

2. **Export and verify locally first.** `litert_torch.signature("step",
   debug_wrapper, args).convert()` to a throwaway `.tflite`, then confirm
   via `ai_edge_litert.interpreter.Interpreter` that every tap matches the
   eager PyTorch value (`nan=False` everywhere) and that `output_N`'s
   numeric suffix equals the tap's true position in the returned tuple --
   it is **not** alphabetically sorted, despite what
   `get_signature_list()` might suggest. This step only proves conversion
   fidelity (still CPU-side); it does not touch the real bug.

3. **Deploy to the app and run CPU vs GPU on-device.** A minimal temporary
   debug harness in the app:
   - An object that mmaps the debug model, loads seeded input `.bin`
     files, and runs the interpreter **once per backend, logging and
     discarding each result before starting the next** -- holding CPU's
     and GPU's full output sets simultaneously is a real OOM risk with
     this model's tensor sizes, even with `largeHeap`.
   - Run three configs, not two: CPU, GPU with default
     `GpuDelegateFactory.Options()`, and GPU with
     `setPrecisionLossAllowed(false)`. The third leg is what separates bug
     class 1 from class 2 above, for free, no reconversion needed.
   - Log per-tensor `min`/`max`/`mean`/`nan`/`inf` (skip NaN/Inf values
     when computing min/max so a partially-corrupted tensor still shows a
     finite range for its clean elements -- that partial-vs-total pattern
     is itself a clue: a 1-in-N corruption fraction points at a specific
     cached/sequence position, a 100%-corrupted tensor points upstream of
     any per-position split).
   - Trigger it from the app's entry point (guarded by an
     `AtomicBoolean` so activity recreation can't double-run it and
     double the memory footprint), with `android:largeHeap="true"` in the
     manifest (GPU delegate init alone can use 500+MB) and the real
     model assets moved aside so the debug APK stays small enough to
     install on a storage-constrained device.
   - `adb logcat -s YourTag:*` and grep for `nan=true`/`inf=true`.

4. **Bisect by moving the tap points.** Compare CPU vs GPU-default at each
   tap, working from a known-clean boundary (an input already confirmed
   clean from a previous round) toward the first tap that shows
   `nan=true`/wildly-diverging values. Re-instrument narrower each round
   -- e.g. module input clean, module's first internal op's output
   already corrupted → the bug is in that first op, not downstream.

5. **Never trust a fix without re-running this loop end-to-end.** A
   candidate op-level rewrite that produces **byte-identical** on-device
   GPU output to before is proof the rewritten op was never the cause --
   this happened three times here (a `SLICE`→`index_select` swap, a
   `baddbmm`-with-uninitialized-`torch.empty` rewrite, and an initial
   `GroupNorm` reduction-axis change all had zero measured effect and
   were reverted). That is not wasted effort: it's a real result that
   narrows the search, but only if you actually check the *numbers*
   on-device, not just whether `compare.py` still passes (it will --
   see the note at the top of this skill).

## Cleanup, every round

Debug scaffolding must be fully removed before moving on, and verified
removed, not assumed:

```bash
# In the app repo
rm NodeDiagnostic.kt-equivalent  # or whatever the debug harness file was
# remove its one-line trigger + imports from the entry-point Activity
# remove android:largeHeap="true" from AndroidManifest.xml
# move the real model assets back from wherever they were parked
git status --short   # must exactly match the pre-investigation baseline
git diff --stat
```

```bash
# In this repo
rm -rf tflite_models/gpu_debug/   # or wherever debug exports landed
```

A leftover debug trigger, manifest flag, or missing model asset is a
regression in its own right, independent of whether the actual bug got
fixed.

## Pitfalls specific to this loop

- **Gradle can serve a stale cached APK** with the old debug model/assets
  still bundled unless built with `--rerun-tasks`. Always check the built
  APK's size (`ls -la app-debug.apk`) against what it should be before
  installing -- a debug-only build and a build with the real ~250MB+
  model assets still bundled differ by hundreds of MB, an easy tell.
- **`adb uninstall` before every install** on a storage-constrained
  device, and watch for stale files under `/data/local/tmp/vda_benchmark`
  left by earlier `benchmark_gpu.sh` runs eating the same limited space.
- **A partially-corrupted tensor can still show a plausible-looking
  `min`/`max`.** Always check the `nan`/`inf` flags explicitly, not just
  whether the range looks sane -- a 1-in-8 corrupted cache position can
  hide inside an otherwise-normal-looking min/max.
