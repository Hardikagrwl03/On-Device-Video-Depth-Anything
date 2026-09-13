---
name: vda-app-verify
description: Verify VDA app changes on a real device with adb -- driving the UI from the shell, reading per-stage logs, confirming downloads, and above all confirming a depth run produced a real depth map rather than a flat one. Use when testing a change end-to-end, reproducing a crash, measuring per-frame timing, or checking GPU output against CPU rather than assuming a build success means it works.
---

# VDA app: on-device verification

A green build proves nothing here. The app once launched cleanly, downloaded,
ran every frame, reported `Done`, and wrote a black video -- the model's
output was a constant and nothing in the build or logs said so. Everything
below is `adb`-driven so it can run without touching the phone.

## Logging tags

```bash
adb logcat -c
adb logcat -d | grep -E "Controller|DepthModule|TFLiteModelRunner|ModelRepository|ModelDownloader"
```

| Tag | Covers |
|---|---|
| `Controller` | per-frame timing of the decode/infer/render/encode pass |
| `DepthModule` | config summary, `VDA init executed` (once per run) / `VDA step executed` (per frame) |
| `TFLiteModelRunner` | `loadModel: Using GPU/CPU`, **delegate-rejection CPU fallback**, inference timing |
| `ModelRepository` / `ModelDownloader` | bootstrap, per-file start/complete (elapsed, MB/s), failure stack traces |
| `VideoFrameDecoder` / `VideoFrameEncoder` | decode/encode per frame |

Prefer `adb logcat -d | grep` over `-s TAG`. To block until an event:

```bash
timeout 600 adb logcat -v brief | grep -m1 -E "depthVideo: Frame 19[0-9] "
```

## Driving the UI from the shell

Screenshots are the reliable way to know where you are; taps on a stale
assumption silently hit the wrong thing (a picker's grid reorders whenever the
gallery changes). Dump the tree when you need coordinates:

```bash
adb exec-out screencap -p > shot.png
adb shell uiautomator dump >/dev/null 2>&1 && adb shell cat /sdcard/window_dump.xml | grep -oE 'text="[^"]*"'
```

Two dialogs commonly block a fresh install: the lock screen (`input keyevent
KEYCODE_WAKEUP`, swipe up) and a "This app isn't 16 KB-compatible" platform
warning about TFLite's `.so` alignment -- pre-existing and unrelated; dismiss it.
It appears only on debug builds.

## What to actually assert

- **Downloads**: both files present at their **exact manifest byte sizes**, no
  `.part` left behind: `adb shell run-as dev.hamster.vda ls -la files/models`.
  Then `ModelRepository: ... installed` **and** `TFLiteModelRunner: loadModel:
  Using ...` without a relaunch -- the ViewModel must configure itself.
- **Model load**: `DepthModule: Model:` names the file you expected,
  `Device:` the device you expected (it can be coerced to CPU -- see
  `vda-app-models`).
- **Depth run**: the completion strip reads `Done in Xs · Y ms/frame avg`.
  Healthy on a Galaxy S23 FE, 1280x720, `gpu` source, FP32: **~2.2 s/frame**
  at the start, climbing to ~4.5 s as the SoC throttles (195-frame clip
  ≈ 11 min, 3.4 s/frame average); `init` costs ~7 s once. CPU is ~8.5 s/frame.
- **The output is not flat.** This is the one that matters. Look at the
  preview, and look at the file sizes: a 195-frame depth video that is ~11 KB
  is a constant frame. A real one is megabytes. Switch to Colormap too --
  near must be yellow/orange, far purple/black; cyan means the encoder's
  channel order regressed.
- **Crashes**: `adb logcat -d | grep -c "FATAL EXCEPTION"` should be `0`.

## Proving GPU output numerically (the test that found the bug)

Delegation coverage, `verify.py`, and a clean run all passed while the GPU
returned a constant. The check that catches it is a CPU-vs-GPU output dump
with the stock `benchmark_model` binary (in the converter repo under
`benchmark/binary/`), no app involved:

```bash
B=/data/local/tmp/android_aarch64_benchmark_model
adb push <init.tflite> /data/local/tmp/m.tflite
adb shell "$B --graph=/data/local/tmp/m.tflite --num_runs=1 --warmup_runs=0 --output_filepath=/data/local/tmp/cpu.bin"
adb shell "$B --graph=/data/local/tmp/m.tflite --num_runs=1 --warmup_runs=0 --use_gpu=true --gpu_precision_loss_allowed=false --output_filepath=/data/local/tmp/gpu.bin"
adb pull /data/local/tmp/cpu.bin; adb pull /data/local/tmp/gpu.bin
python3 -c "
import numpy as np
for t in ['cpu','gpu']:
    a=np.fromfile(t+'.bin',dtype=np.float32,count=720*1280)
    print(t,'unique',np.unique(a).size,'nan',np.isnan(a).sum(),'range',a.min(),a.max())"
```

The dump concatenates all nine outputs (depth first, then the eight caches);
slice by the shapes in `DepthHiddenStates`. Healthy: identical unique counts
within a few hundred, max |diff| ~1e-4, zero NaN. Broken looked like
`gpu unique 1` for depth and 100% NaN in the two largest caches. Always pass
`--gpu_precision_loss_allowed=false` -- that is what the app ships.

## Side-loading models (network-free testing)

The app's store is plain files, so a pair can be installed without the
network -- note the rename from the converter's naming to the release naming:

```bash
adb push vda_vits_720x1280_input518_infer8_init.tflite /data/local/tmp/init.tflite
adb shell "cat /data/local/tmp/init.tflite | run-as dev.hamster.vda sh -c 'cat > files/models/vda_gpu_vits_720x1280_input518_infer8_init.tflite'"
```

(`run-as` cannot read `/data/local/tmp` directly; the pipe works.) The sizes
must match the manifest exactly or the store reports the pair absent.

## Fresh-install and offline testing

`adb shell pm clear dev.hamster.vda` wipes `filesDir`, so the pair re-downloads
-- the only way to genuinely exercise bootstrap. A release-signed APK cannot
be installed over a debug one; `pm uninstall` between them.

**Airplane mode is not enough** on Samsung (Wi-Fi survives it). Disable the
radios explicitly: `adb shell svc wifi disable && adb shell svc data disable`;
re-enable with `enable`.

## Screenshots for the docs

`docs/images/*.png` are 1080x2340 captures with the ~95 px status bar cropped
and scaled to 280 px wide:

```python
from PIL import Image
im = Image.open("shot.png").convert("RGB"); w, h = im.size
im = im.crop((0, 95, w, h)); im = im.resize((280, round(280 * (h - 95) / w)), Image.LANCZOS)
im.save("docs/images/name.png", optimize=True)
```

Use the converter's public example clip for output shots, never a personal
gallery video.
