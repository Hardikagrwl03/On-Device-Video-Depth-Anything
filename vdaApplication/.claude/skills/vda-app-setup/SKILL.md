---
name: vda-app-setup
description: Get the VDA Android app building and running after a fresh clone -- SDK/JDK requirements, supported ABIs, and how the model pair arrives at runtime. Use when someone has just cloned vdaApplication, is setting up a device or emulator, or hits errors like "Model not downloaded", an INSTALL_FAILED_NO_MATCHING_ABIS, or an empty Models page.
---

# VDA app: first-time setup

An Android app that runs Video Depth Anything on-device. A fresh clone builds
and runs with **no manual asset step** -- there is nothing to copy into
`app/src/main/assets/`. That directory does not exist and must not be
recreated (an earlier version bundled ~460 MB of models there and shipped a
683 MB APK); models are downloaded at runtime instead (see `vda-app-models`).

## Requirements

- Android Studio (Narwhal or newer) with the Android SDK, and **JDK 11**.
- A device or emulator on **API 35+** (`minSdk = 35`, `targetSdk = 37`,
  `compileSdk = 37`).
- The device must be **`arm64-v8a`**, or the emulator **`x86_64`**. Those are
  the only ABIs packaged -- see "ABI" below.
- Gradle 9.5 (via the wrapper), AGP 9.3.1, Kotlin 2.2.10.
- For a GPU run, a phone: the delegate's OpenCL path is what makes VDA usable
  (~2 s/frame vs ~8 s on CPU). An emulator will run, on CPU, very slowly.

```bash
./gradlew --offline installDebug     # --offline once the cache is warm; drop it on first sync
adb shell am start -n dev.hamster.vda/.MainActivity
```

## What happens on first launch

`MainActivity.onCreate` calls `ModelRepository.ensureBootstrapModels()`, which
downloads **one model pair** from the `models-v1` GitHub release into
`filesDir/models/`:

1. `vda_gpu_vits_720x1280_input518_infer8_init.tflite` (~121 MB)
2. `vda_gpu_vits_720x1280_input518_infer8_step.tflite` (~124 MB)

A VDA model is always a pair -- `init` seeds the temporal caches from the
first frame, `step` runs every later frame -- so the first run needs
**network** and pulls ~246 MB before anything works. There is no small model
that lands early, unlike some other apps of this shape. Until the pair is
installed the home screen's Video Depth tile shows download progress; if the
download failed it reads "Tap to download a model" and routes to the Models
page. Nothing crashes offline.

To watch it:

```bash
adb logcat -d | grep -E "ModelRepository|ModelDownloader"
adb shell run-as dev.hamster.vda ls -l files/models
```

The app configures itself the moment the pair lands -- `DepthViewModel`
observes `ModelRepository.states` -- so no restart is needed.

## ABI

`app/build.gradle.kts` sets `ndk { abiFilters += listOf("arm64-v8a", "x86_64") }`.
TFLite's native libraries are ~70 MB **per ABI**, and at `minSdk 35` nothing
can reach `armeabi-v7a` or `x86` (no Android 15 device ships 32-bit-only ARM,
and x86 Android phones do not exist).

Consequence: a 32-bit-only emulator image fails to install with
`INSTALL_FAILED_NO_MATCHING_ABIS`. Create an `x86_64` (or arm64) AVD instead.

## A test clip

The model accepts **exactly 1280x720 landscape** input and refuses anything
else with a message naming both sizes. The converter repo ships a suitable
public clip: `Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4`
(1280x720, 195 frames). Push it and let the media scanner see it:

```bash
adb push Tokyo-Walk_rgb.mp4 /sdcard/Movies/
adb shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Movies/Tokyo-Walk_rgb.mp4
```

## What is gitignored and absent from a clone

- `*.tflite` -- models are never committed and never bundled.
- `vda-release.keystore`, `keystore.properties` -- release signing material.
  Release builds still work without them, just unsigned (see `vda-app-release`).
- `dist/` -- staged APKs.
- `local.properties` -- your SDK path; Android Studio regenerates it.

## Where to read next

- `README.md` -- architecture, project structure, logging tags.
- `USER_GUIDE.md` -- what the app does from the user's side.
- The converter's README (branch `converter`, `vdaConverter/README.md`) -- how
  the `.tflite` pairs are produced, and the GPU-delegate correctness story
  every app developer here should know (see `vda-app-models`).
