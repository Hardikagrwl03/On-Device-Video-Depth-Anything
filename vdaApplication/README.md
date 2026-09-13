# Application for On-Device Video Depth Anything

An Android application that runs [Video Depth Anything (VDA)](https://github.com/DepthAnything/Video-Depth-Anything) fully on-device via TensorFlow Lite / LiteRT to produce a temporally consistent depth map for a user-selected video — rendered both as grayscale and on the inferno colormap — with GPU/NNAPI acceleration where available.

The APK contains no model weights. On first launch the app downloads the default model pair (~246 MB) over the network, so **use Wi-Fi the first time**; it becomes usable as soon as that pair lands.

> **Just want to use the app?** See **[USER_GUIDE.md](USER_GUIDE.md)** — installing, first run,
> and every setting explained without reference to the code. The rest of this README is for
> developers working on the app itself.

| Home | Colormap output | Models |
| :---: | :---: | :---: |
| ![Home screen](docs/images/home.png) | ![Colormap output](docs/images/output-colormap.png) | ![Models page](docs/images/models.png) |

## Overview

The user picks a video from their device, the app decodes it frame-by-frame, runs each frame through an on-device VDA TFLite model — carrying the model's eight temporal-attention caches between frames, which is what makes the depth stable from one frame to the next — and re-encodes the result into two output videos in a single decode/inference pass:

- **Grayscale** — the normalised depth as a luminance map (near is bright, far is dark), and
- **Colormap** — the same depth on the inferno ramp VDA's own demo uses (near is yellow, far is deep purple).

Both play back in-app, kept in sync with the input video, and can be saved to the device gallery in one tap.

VDA is a *streaming* model, and it ships as two graphs rather than one. The `init` graph takes the first frame of a sequence alone and seeds the caches; the `step` graph takes every later frame plus the caches and returns the depth map and the updated caches. The app hides that split behind a single module: the first frame after a run starts (or after a reset) goes through `init`, every other frame through `step`, and the `init` interpreter is released once it has done its one job.

### Measured performance

On a Galaxy S23 FE (Snapdragon 8 Gen 1), 720x1280 input, `gpu` source, ViT-S:

| Compute device | Per frame (steady state) | 13 s clip (195 frames) |
| --- | --- | --- |
| GPU delegate, FP32 | ~2.2 s at the start, rising to ~4.5 s as the SoC throttles | ~11 min (3.4 s/frame average) |
| CPU (XNNPack, 4 threads) | ~8.5 s | ~28 min |

Seeding the caches with the `init` graph costs a further ~7 s once per run. An `original`-source model is forced onto the CPU and is no faster than the `gpu` build there; it exists for comparison against the `gpu` rewrites, not for production use.

### Known limitations

These are real constraints of the current pipeline, not bugs to be surprised by:

- **Inputs must be exactly 720x1280.** The exported graph has fixed input/output shapes, and there is only one published resolution. A clip of any other size is refused up front with a message naming both sizes, rather than overflowing a buffer deep inside the decoder.
- **Outputs carry no audio track.** `VideoFrameEncoder` writes video only.
- **Depth is normalised per frame, not per clip.** Upstream normalises over the whole video before rendering; a single streaming pass can't know the clip's range in advance without buffering every frame's depth. The trade-off is some brightness flicker when a scene's depth range changes shot to shot.
- **Model downloads do not resume mid-file.** A transfer killed with the process restarts that file from zero — though a pair interrupted between its two halves resumes at the file boundary.
- **Live Depth is not implemented** — the tile shows a "coming soon" toast.

## Features

- On-device video depth estimation using VDA — inference is entirely local; the network is used only to fetch model weights.
- Models are **downloaded on demand** from the [`models-v1` GitHub release](https://github.com/Hardikagrwl03/On-Device-Video-Depth-Anything/releases/tag/models-v1) rather than bundled in the APK, with an in-app Models page to browse and install both published pairs; the `gpu` pair is fetched automatically on first launch.
- Selectable compute backend per run: GPU delegate, NNAPI (NPU), CPU, or automatic fallback (NNAPI → GPU → CPU).
- Two converter sources (`gpu`/`original`) and the converter's own resolution / input-size / temporal-window parameters, each resolving to a matching `.tflite` pair actually **downloaded** into `filesDir/models/` (the config UI can't offer a combination that isn't installed).
- Frame-accurate passing of VDA's eight temporal-attention caches between inference calls, so depth stays consistent across frames instead of being re-estimated from scratch.
- One decode/inference pass produces both renderings — grayscale and inferno — switchable in-app and played back in sync with the input; both can be saved to `Movies/VDA` in the gallery.
- Native Jetpack Compose UI: a home screen, a non-scrolling depth screen with synchronized Media3 input/output preview playback, a scrollable model-settings sheet, and Save-to-gallery.
- Every stateful native resource — the TFLite interpreters and GPU delegate, each `MediaCodec`/`MediaMuxer` encoder, the `MediaMetadataRetriever` decoder — is confined to its own dedicated thread, since none of them are safe to drive from more than one thread (the GPU delegate in particular binds an EGL context to whichever thread builds it).

The app uses a fixed brand colour scheme (not Material You dynamic colour) so its look is consistent across devices, and because its own content — a grayscale map and a fixed scientific colormap — should never be re-tinted by a wallpaper. The palette is a set of generated tonal ramps at CIE LCh hue 84 (mustard), verified for WCAG contrast on every pair the light and dark schemes use; see `ui/theme/Color.kt`.

## Requirements

- Android Studio (Narwhal or newer recommended) with the Android SDK.
- JDK 11.
- An Android device or emulator running **API 35 (Android 15) or newer** (`minSdk = 35`, `targetSdk = 37`, `compileSdk = 37`).
- An **`arm64-v8a`** device, or an **`x86_64`** emulator. Those are the only two ABIs packaged: TFLite's native libraries are ~70 MB *per ABI*, and at `minSdk 35` nothing can reach the other two (no Android 15 device ships 32-bit-only ARM, and there are no x86 Android phones), so `armeabi-v7a` and `x86` are excluded via `ndk { abiFilters }` — see `app/build.gradle.kts`.
- Gradle 9.5 (fetched automatically via the Gradle wrapper), AGP 9.3.1, Kotlin 2.2.10.

## Getting Started

1. **Clone the repository** and open the `application/vdaApplication` folder in Android Studio.
2. **Sync and build** — Android Studio will resolve dependencies (TensorFlow Lite, LiteRT, Jetpack Compose, Media3, AndroidX) automatically via Gradle.
3. **Run** on a device/emulator meeting the API level requirement above. No model files are needed at build time — the app downloads what it needs on first launch (see below).

### How the TFLite models get onto the device

No `.tflite` file is committed to this repository or bundled into the APK. The app fetches them at
runtime from the [`models-v1` GitHub release](https://github.com/Hardikagrwl03/On-Device-Video-Depth-Anything/releases/tag/models-v1)
into internal storage at `filesDir/models/`, where they stay installed.

- **On first launch** the app automatically downloads the `gpu`-source ViT-S pair (`init` ~121 MB,
  then `step` ~124 MB). Every pair is ~240 MB, so there is no small model to make the app usable
  early; it fetches one, and fetches the one that can run on every compute device.
- **The Models page** (from the home screen) lists both published pairs with the converter config
  each was created with, and installs either on tap. Installed entries are marked with an accented
  card and a check.

A "model" here is always a **pair** of files. The models page shows one card per pair, the two
files download back to back behind one progress bar, and the store reports a pair installed only
when both halves are present at their exact size — neither is usable without the other.

Downloads run one at a time on a process-wide scope, so they survive navigating away from the
Models page — but **not** the process dying; there is no mid-file resume, so a killed transfer
restarts that file.

Filenames are the release asset names verbatim, which is also what `DepthConfig` resolves to and
what `ModelStore` looks up — one string, no mapping layer:

```
vda_<source>_<backbone>_<height>x<width>_input<inputSize>_infer<inferLen>_<init|step>.tflite
```

- `<source>`: `gpu` or `original` — **which copy of the VDA PyTorch source the model was traced
  from**, a build-time property. `gpu` builds rewrite every op the TFLite GPU delegate can't run
  (and one it runs *wrongly* — see the note below); `original` is the unmodified upstream graph.
  This is *not* the same thing as the compute device selected at runtime.
- `<backbone>`: `vits` — the only VDA encoder published so far (`vitb`/`vitl` would slot into the
  same pattern)
- `<height>x<width>`: input resolution — every published build is `720x1280`
- `<inputSize>`: the converter's `--input-size`, the short side VDA resizes to internally before
  the ViT (`518`); the working resolution for 720x1280 is 518x924
- `<inferLen>`: the converter's `--infer-len`, the temporal window in frames (`8`), of which
  `infer - 1` are the cached frames each step attends over

`DepthConfig()` (the default) resolves to the `vda_gpu_vits_720x1280_input518_infer8` pair. If that
pair isn't installed yet but another is, `DepthViewModel` configures the installed one instead
rather than failing.

Since the release publishes no checksums, `ModelManifest` records each asset's exact byte size and
`ModelDownloader` verifies both the server's `Content-Length` and the bytes written against it,
downloading into a `.part` file that is renamed only once the check passes. **If a `models-v1`
asset is ever re-uploaded, `ModelManifest.ALL` must be updated in the same commit** or every
install will reject the new file as a size mismatch.

> **`source` vs. compute device.** `source` picks *which file* to load; the compute device
> (`CPU`/`GPU`/`NPU`/`AUTO`) picks *which delegate* runs it. They're independent, and both are
> selectable in the config sheet — but not every pairing can run. An `original` model still
> contains the ops the `gpu` build rewrites, and a delegate is mandatory once attached: rather than
> running just those ops on CPU, the GPU/NNAPI delegate refuses the whole graph and the interpreter
> fails to build. So applying `original` with `GPU` or `NPU` **forces the compute device to CPU**
> and says so in a toast. `AUTO` is left alone — falling back through NNAPI → GPU → CPU is exactly
> what it's for — and `TFLiteModelRunner` still catches a delegate rejection and degrades to CPU as
> a backstop.
>
> **Fully delegated is not the same as correct.** An earlier `gpu` export ran 100% on the GPU
> delegate with no error and returned a *constant* depth map — a black video — for any input. Two
> independent bugs caused it, and both had to be fixed for GPU output to be right: the delegate's
> `MEAN` kernel mis-reduces the `axis=[0, 2]` pattern `nn.LayerNorm` lowers to (fixed in the
> converter's `gpu` source), and the delegate's default FP16 produces NaN in the motion modules
> (fixed here — `TFLiteModelRunner` builds the delegate with `setPrecisionLossAllowed(false)`,
> which no `.tflite` file can enforce on its own). Neither shows up in delegation coverage or in
> the converter's host-side checks. If a future export ever goes flat, check the output's variance
> before believing "fully delegated". The full account is in the
> [converter's README](https://github.com/Hardikagrwl03/On-Device-Video-Depth-Anything/blob/converter/vdaConverter/README.md).

## Building a release APK

Release builds are signed with a keystore that is **deliberately not in version control**
(`.gitignore` excludes `*.keystore`, `*.jks` and `keystore.properties`). `app/build.gradle.kts`
reads `keystore.properties` from the project root:

```properties
storeFile=vda-release.keystore
storePassword=...
keyAlias=vda-release
keyPassword=...
```

If that file is absent the project still builds — `assembleRelease` simply produces an **unsigned**
APK — so a fresh clone isn't blocked by a missing secret. Note that a PKCS12 keystore (keytool's
default) has no separate key password: `keyPassword` must equal `storePassword`, or packaging
fails with "Given final block not properly padded".

```bash
./gradlew assembleDebug assembleRelease
# app/build/outputs/apk/debug/app-debug.apk
# app/build/outputs/apk/release/app-release.apk
```

Both are universal APKs carrying `arm64-v8a` + `x86_64` (see Requirements), so a single file can be
installed by any supported device. Signing uses the v3 scheme (v1/v2 are redundant at `minSdk 35`,
and v3 is what permits key rotation later). Verify with:

```bash
$ANDROID_HOME/build-tools/<version>/apksigner verify -v app/build/outputs/apk/release/app-release.apk
```

Stage the artifacts with release naming — the release APK carries **no `-release` suffix**; only
non-default variants are qualified:

```bash
mkdir -p dist
cp app/build/outputs/apk/release/app-release.apk dist/vda-<version>.apk
cp app/build/outputs/apk/debug/app-debug.apk     dist/vda-<version>-debug.apk
sha256sum dist/*.apk
```

The debug build is `debuggable` and signed with the shared Android debug key — fine locally, not
something to hand to users. Remember to bump **both** `versionCode` (must strictly increase for
Android to accept an update) and `versionName` in `app/build.gradle.kts` before building.

> **The keystore is irreplaceable.** Android will only install an update over an existing install if
> it is signed with the same key. If `vda-release.keystore` is lost, every existing user has to
> uninstall before they can take an update. Back it up together with `keystore.properties`
> **outside this repository** — a gitignored file is still destroyed by `git clean -xdf`.

## Project Structure

```
README.md            # this file -- developer-facing
USER_GUIDE.md        # end-user walkthrough
docs/
└── images/          # screenshots used by both documents
.claude/skills/      # six project-scoped Claude Code skills (see below)
```

```
app/src/main/java/dev/hamster/vda/
├── MainActivity.kt                    # Entry point: hosts the home/depth/models screen switch, wires up DepthViewModel
├── Controller.kt                      # Orchestrates VideoFrameDecoder + 2x VideoFrameEncoder + DepthModule end-to-end
├── ui/                                # Jetpack Compose UI
│   ├── HomeScreen.kt                  #   Landing screen: logo, tagline, Video Depth / Live Depth / Models tiles
│   ├── DepthScreen.kt                 #   The depth screen: configure pill, input/output previews, action bar
│   ├── DepthViewModel.kt              #   DepthUiState + all screen logic; owns the Controller
│   ├── ConfigSheet.kt                 #   Bottom sheet for editing DepthConfig (device/source/resolution/backbone/input size/window/threads)
│   ├── ModelsScreen.kt                #   Browse and install the published model pairs; shows each one's converter config
│   ├── theme/                         #   Fixed brand colour scheme, type scale, shape scale
│   │   ├── Theme.kt                   #     VdaTheme - builds the light/dark ColorScheme from Color.kt
│   │   ├── Color.kt                   #     The brand palette's tonal ramps (mustard)
│   │   ├── Type.kt                    #     Typography overrides (wordmark display style, etc.)
│   │   └── Shape.kt                   #     Corner-radius scale for cards/sheets/surfaces
│   └── player/                        #   Media3 ExoPlayer wrapper for synchronized input/output playback
│       ├── DualVideoSync.kt           #     Leader/follower playback sync (drift correction via speed nudging)
│       └── VideoSurface.kt            #     TextureView-backed Compose video surface
├── interfaces/                        # Generic contracts every inference module implements
│   ├── ModuleInterface.kt             #   configure/run/reset/close, generic over a module's Config and IO types
│   ├── ConfigInterface.kt             #   minimal shape every module config resolves to (height, width, RuntimeConfig)
│   └── HiddenStatesInterface.kt       #   contract for a module's recurrent/hidden-state buffers
├── depth/                             # VDA depth module (implements the interfaces above)
│   ├── DepthModule.kt                 #   runs the init/step pair; confined to its own thread (see utils/ConfinedRunner.kt)
│   ├── DepthIO.kt                     #   DepthModule's named input/output buffers
│   ├── DepthConfig.kt                 #   source/resolution/backbone/input-size/window settings; derives both model filenames and the working resolution
│   └── DepthHiddenStates.kt           #   the 8 temporal-attention caches VDA passes between frames
├── models/                            # On-demand model download and the on-disk model store
│   ├── ModelManifest.kt               #   ModelSource + ModelSpec (one init/step pair) + the static table of release assets
│   ├── ModelStore.kt                  #   filesDir/models/: which pairs are installed, length-checked per file
│   ├── ModelDownloader.kt             #   HttpURLConnection streaming into a .part file, size-verified, one file at a time
│   ├── ModelRepository.kt             #   Process-wide download state (StateFlow) + serial queue; downloads a pair as one unit
│   └── ModelCatalog.kt                #   narrows installed pairs into valid config combinations
├── modelRunner/
│   ├── TFLiteModelRunner.kt           #   module-agnostic TFLite Interpreter/delegate wrapper; GPU delegate built FP32-only
│   ├── ModelRunnerInterface.kt        #   the contract TFLiteModelRunner implements
│   └── RuntimeConfig.kt               #   model filename + compute device + thread count
├── video/                             # Video decode/encode, one class per direction, each on its own thread
│   ├── VideoFrameDecoder.kt           #   MediaMetadataRetriever-based frame extraction
│   ├── VideoFrameDecoderInterface.kt
│   ├── VideoFrameEncoder.kt           #   MediaCodec/MediaMuxer-based frame re-encoding (NV12)
│   └── VideoFrameEncoderInterface.kt
└── utils/
    ├── ConfinedRunner.kt              #   confines an object's work to one dedicated thread
    ├── DepthColormap.kt               #   per-frame depth normalisation; grayscale and inferno renderings
    ├── MediaStoreSaver.kt             #   copies a cache-directory output video into the shared gallery
    └── SharedBuffer.kt                #   native shared-memory-backed ByteBuffer helper
```

## Architecture

Inference modules follow a small, generic pattern so new models can be added without changing the surrounding app code:

- **`ModuleInterface<Config : ConfigInterface, IO>`** — every module implements `configure(config)`, `run(io, count)`, `reset()`, and `close()`. `Config` and `IO` are generic per module: each module defines its own config data class (e.g. `DepthConfig`) and its own named input/output data class (e.g. `DepthIO`), so callers get readable, typed fields instead of positional buffers, while the app can still drive any module polymorphically.
- **`ConfigInterface`** — the minimal shape a config must expose (`height`, `width`, a `RuntimeConfig`) so the module can resolve and load the right `.tflite` model. `DepthConfig` carries a second `RuntimeConfig` for the `init` graph, derived from the first so a single device/thread choice covers both, and re-derives the model's internal working resolution (518x924 for 720x1280) the same way the converter does, because the cache shapes depend on it and nothing in the file names records it.
- **`ModelManifest` / `ModelStore` / `ModelDownloader` / `ModelRepository`** — the model-delivery layer. The manifest is the single source of truth for what a valid model is (a `.tflite` in the store that it doesn't list is deliberately invisible); the store owns `filesDir/models/` and reports a pair installed only when both files match the manifest exactly; the downloader streams one release asset into a `.part` file and renames it only after verifying the size, and the repository sequences a pair's two files as one install. `ModelRepository` is a process-wide singleton rather than a ViewModel, because a 240 MB download has to survive navigating away from the Models page — a `viewModelScope` would cancel it. Downloads run one at a time behind a `Mutex`, with a distinct `Queued` state so a waiting entry never renders as a stalled 0%.
- **`HiddenStatesInterface` / `DepthHiddenStates`** — the contract for a module's recurrent state buffers (reset/rewind/put/close). VDA's states are not feature maps but *token* caches: eight tensors shaped `[1, tokens, context, channels]`, where the token grid is the ViT patch grid at the working resolution, scaled per DPT decoder stage. At 518x924 with a context of 7 they total ~80 MB, so they're zeroed a chunk at a time rather than through a scratch array the size of the buffer.
- **`ModelRunnerInterface` / `TFLiteModelRunner`** — the shared, module-agnostic wrapper around a TFLite `Interpreter`: (re)builds the interpreter and delegate (GPU/NNAPI/CPU/AUTO) only when the resolved `RuntimeConfig` actually changes, and exposes `run`/`close` for single- and multi-tensor inference. Every call is confined to one dedicated thread via `ConfinedRunner`, so the interpreter is always built and invoked on the same thread — required for the GPU delegate, whose EGL context is bound to its creating thread. The GPU delegate is always built with FP16 disabled (see the correctness note above). It memory-maps the model out of `ModelStore` (never downloads — that would block the confined thread on unbounded network I/O), and if the selected delegate refuses the graph it degrades to CPU rather than letting the constructor's exception kill the app.
- **`DepthModule`** — the concrete depth implementation. It owns *two* `TFLiteModelRunner`s. The first `run` of a sequence goes to the `init` runner, which seeds the caches from the frame alone; every later `run` combines the caller's frame/depth buffers (`DepthIO`) with the caches, invokes the `step` runner, and copies the freshly produced caches back in for the next call. The `init` interpreter is closed as soon as it has seeded — it's ~120 MB of weights nothing needs until the next `reset()` — and rebuilt on demand. `reset()` zeroes the caches *and* re-arms `init`: zeroed caches are not a valid sequence start for VDA. Also confined to its own dedicated thread.
- **`VideoFrameDecoder` / `VideoFrameEncoder`** — a `MediaMetadataRetriever`-based frame extractor and a `MediaCodec`/`MediaMuxer`-based frame encoder, each confined to its own thread (`MediaCodec`/`MediaMuxer`/`MediaMetadataRetriever` aren't safe to drive from more than one thread). `Controller` owns one decoder and two encoders (grayscale, colormap), so both can write a separate output video from the same decode/inference pass. The encoder packs frames as NV12 — the layout `COLOR_FormatYUV420SemiPlanar` actually names — reading each pixel's floats as R,G,B; an earlier version wrote NV21 chroma order and read B,G,R, two swaps that cancel for RGB input but inverted the inferno ramp.
- **`DepthColormap`** — turns the model's raw output (unbounded relative inverse depth, larger = nearer) into something encodable: per-frame `(d - min) / (max - min)` into 0–255, then either straight to grayscale or through a 12-stop piecewise-linear approximation of matplotlib's `inferno`.
- **`Controller`** — the app-level orchestrator: loads the model pair, decodes the input video frame-by-frame, runs each frame through `DepthModule`, renders the depth both ways, and writes both outputs via their respective encoders — one decode/inference pass, two videos out. It refuses a clip whose dimensions don't match the configured model up front, with a message naming both sizes.
- **`DepthViewModel`** — owns the `Controller` and exposes a single `DepthUiState` `StateFlow` for the depth screen. Runs `Controller.configure()` (which can rebuild the GPU delegate — several seconds for a ~120 MB graph) off the main thread, so the UI can show a spinner instead of appearing to hang; the first navigation into the depth screen triggers this same path, since `DepthViewModel` is created lazily on first use. It coerces the compute device to CPU when the selected model can't run on the chosen delegate (see the `source` note above), and resolves the config against what's actually installed: if the configured pair isn't downloaded yet but another is, it configures that one instead, and if nothing is installed it reports `modelMissing` and configures automatically as soon as a download lands.
- **`ui/theme`** — a fixed brand colour scheme (not Material You dynamic colour), so the app looks the same regardless of the device's wallpaper.
- **`ui/player`** — `DualVideoSync` drives a leader (input) and follower (output) Media3 `ExoPlayer` pair, keeping them aligned via small playback-speed nudges rather than continuous re-seeking; `VideoSurface` renders each into a plain `TextureView` so Compose can clip, round, and cross-fade it.

## Usage

For an end-user walkthrough with screenshots, see **[USER_GUIDE.md](USER_GUIDE.md)**. In brief:

| Depth screen | Ready to run |
| :---: | :---: |
| ![Depth screen](docs/images/depth-empty.png) | ![Ready to run](docs/images/ready.png) |
| **Run in progress** | **Grayscale output** |
| ![Run in progress](docs/images/running.png) | ![Grayscale output](docs/images/output-grayscale.png) |
| **Colormap output** | **Model settings** |
| ![Colormap output](docs/images/output-colormap.png) | ![Model settings](docs/images/config-sheet.png) |

1. Launch the app. On a fresh install it starts downloading the default model pair straight away; the **Video Depth** tile shows that progress and becomes available as soon as the pair lands. Tap **Models** to browse and install either published pair, or **Live Depth** for a "coming soon" toast — real-time camera depth isn't implemented yet.
2. Tap **Video Depth**.
3. Tap **Import** to pick a 720x1280 video from the device.
4. Tap **Depth** to run depth estimation over every frame; a status strip shows progress, and **Cancel** stops the run.
5. Once done, switch between **Grayscale** and **Colormap** — both played back in sync with the input — and tap **Save** to copy both to `Movies/VDA` in the gallery.
6. Tap **Reset** to clear the current selection and start over.

The **Model** pill opens the settings sheet: compute device, source, resolution, backbone, input size, temporal window and thread count. Every dropdown is populated from the pairs actually installed, and the sheet shows both file names the selection resolves to. Applying a change rebuilds the interpreter; the sheet stays open with a spinner until that finishes.

## Logging

Every component logs through `android.util.Log` under its own tag, so `adb logcat -s <TAG>` (or the equivalent filter in Android Studio's Logcat panel) can isolate just the component you're interested in:

| Tag | Source | Covers |
| --- | --- | --- |
| `Controller` | `Controller.kt` | End-to-end orchestration: per-frame timing for the decode/infer/render/encode pass |
| `DepthModule` | `depth/DepthModule.kt` | Depth configuration, per-run init/step inference timing, cache reset/close |
| `TFLiteModelRunner` | `modelRunner/TFLiteModelRunner.kt` | Interpreter/delegate setup (GPU/NNAPI/CPU/AUTO), delegate-rejection CPU fallback, per-call inference timing, interpreter close |
| `ModelRepository` | `models/ModelRepository.kt` | Bootstrap enqueueing, per-pair install/failure, full stack traces for failed downloads |
| `ModelDownloader` | `models/ModelDownloader.kt` | Per-file download start (URL, expected bytes), completion (elapsed, MB/s), failure |
| `VideoFrameDecoder` | `video/VideoFrameDecoder.kt` | Per-frame decode timing |
| `VideoFrameEncoder` | `video/VideoFrameEncoder.kt` | Per-frame encode timing (one line per encoder - grayscale, colormap - per frame) |
| `testDummyInputs` | `modelRunner/TFLiteModelRunner.kt` | `testDummyInputs()` diagnostics: input/output tensor shapes, dtypes, sizes |
| `logSignature` | `modelRunner/TFLiteModelRunner.kt` | `logSignature()` diagnostics: input/output tensor names, shapes, dtypes |

To follow just the depth pipeline end-to-end, filter on multiple tags at once:

```
adb logcat -s Controller:D DepthModule:D TFLiteModelRunner:D VideoFrameDecoder:D VideoFrameEncoder:D
```

Note: `TFLiteModelRunner.configure()`'s own summary line ("configure: TFLite Model Runner configured with: ...") logs under the `DepthModule` tag rather than `TFLiteModelRunner` — if you're filtering strictly on `TFLiteModelRunner`, that specific line won't show up there; its device-selection and interpreter-lifecycle lines (`loadModel: Using GPU`, `Interpreter Created`, `close: ...`) do use the `TFLiteModelRunner` tag.

To watch which thread each component is running on (`DepthModule`, `VideoFrameDecoder`, and each `VideoFrameEncoder` instance are confined to one dedicated, named thread), look at the TID column in `adb logcat -v threadtime`, or:

```
adb shell ps -T -p $(adb shell pidof dev.hamster.vda) | grep vda-
```

A run that finishes with a **flat output** — every pixel the same value — is the signature of the GPU delegate mis-executing the graph (see the correctness note under "How the TFLite models get onto the device"). The output files give it away too: a 195-frame depth video that is only a few kilobytes is a constant frame, not a depth map.

## Claude Code skills

`.claude/skills/` holds six project-scoped skills (auto-discovered by Claude Code from this repo,
no setup needed) documenting the workflows and invariants in more operational detail than this
README:

- `vda-app-setup` — fresh clone to running app: requirements, supported ABIs, what happens on first
  launch, where to find a test clip, and what's deliberately absent from a clone
- `vda-app-architecture` — the generic module pattern, the init/step two-graph lifecycle, thread
  confinement, and the traps that deadlock, crash, or silently produce wrong output if you edit
  around them
- `vda-app-models` — the on-demand model-pair system end to end, how to add or update a pair, why
  `source` is not the compute device, and why "fully delegated" is not "correct" on the GPU
- `vda-app-ui` — Compose conventions: the generated mustard palette and how to regenerate it,
  navigation, and the card/badge/sheet patterns to copy
- `vda-app-verify` — driving the app from `adb` to verify a change on a real device, including the
  CPU-vs-GPU output comparison that catches a flat depth map, side-loading models, and the
  screenshot recipe for the docs
- `vda-app-release` — signing (and the PKCS12 password trap), ABI packaging, artifact naming, and
  what not to publish

## Adding a New Module

To add a new on-device model (e.g. segmentation):

1. Create a package under `dev/hamster/vda/<yourmodule>/`.
2. Define `<YourModule>Config : ConfigInterface` with whatever settings your model needs, resolving to a `RuntimeConfig` (model filename, compute device, thread count).
3. Define `<YourModule>IO` — a data class with named `ByteBuffer` fields for your model's inputs/outputs.
4. Implement `<YourModule> : ModuleInterface<YourModuleConfig, YourModuleIO>`, using `TFLiteModelRunner` internally the same way `DepthModule` does. If your model is recurrent, implement `HiddenStatesInterface` for its state buffers. Confine the module's work to its own thread with `ConfinedRunner`, the same way `DepthModule` does, if it wraps any native/stateful resource.
5. Add your model's `.tflite` files to `ModelManifest.ALL` (source, backbone, resolution, converter parameters and exact byte sizes) so they can be downloaded and so `ModelCatalog` will offer them — `TFLiteModelRunner` loads only what the manifest lists. If your model is a single file rather than a pair, `ModelSpec` will need a second shape.
6. Wire it into `Controller` (or a new controller) alongside `DepthModule`.
