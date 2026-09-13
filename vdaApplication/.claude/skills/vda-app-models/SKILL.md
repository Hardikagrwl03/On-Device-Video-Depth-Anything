---
name: vda-app-models
description: How the VDA app downloads, stores and selects .tflite model pairs at runtime, how to add or update one, and why "fully delegated" is not "correct" on the GPU. Use when changing ModelManifest/ModelStore/ModelDownloader/ModelRepository, publishing new models to the models-v1 release, or debugging "Model not downloaded", a size-mismatch failure, an empty config sheet, or a black/flat depth output.
---

# VDA app: the on-demand model system

No model is bundled. `app/src/main/assets/` does not exist and must not be
recreated. Models are fetched from the
[`models-v1` GitHub release](https://github.com/Hardikagrwl03/On-Device-Video-Depth-Anything/releases/tag/models-v1)
into `filesDir/models/`.

## A model is a pair

VDA exports two graphs, `init` and `step`, and neither is usable alone. So
`ModelSpec` describes a **pair** -- both file names, both URLs, both sizes --
and that is the unit the manifest lists, the store installs, the repository
downloads, and the Models page renders as one card.

| File | Role |
|---|---|
| `ModelManifest.kt` | `ModelSource`, `ModelSpec`, and the static table of both pairs. **Single source of truth** -- a `.tflite` in the store the manifest does not list is invisible by design. |
| `ModelStore.kt` | `filesDir/models/`. `isFileInstalled` checks **length against the manifest**; `isInstalled(spec)` requires **both** halves. `installedBytes` lets a half-installed pair report real progress. |
| `ModelDownloader.kt` | One file: `HttpURLConnection` -> `<name>.part` -> size check -> rename. Deletes the `.part` on any failure *or cancellation*. |
| `ModelRepository.kt` | Process-wide singleton with `StateFlow<Map<String, ModelDownloadState>>` keyed by `spec.id`, a serial queue, and the loop that downloads a pair's two files under one lock as one progress bar, skipping a half already on disk. |

`ModelCatalog.kt` narrows *installed* pairs into the config sheet's
source -> resolution -> backbone -> input size -> window chain.

## Why `ModelRepository` is a singleton, not a ViewModel

A 240 MB download must survive navigating off the Models page; a
`viewModelScope` would cancel it. Screens collect `states` directly. Downloads
run **one at a time** behind a `Mutex`; waiting entries report `Queued`.

Known limitation: a download dies with the process and cannot resume
**mid-file**. It does resume at the file boundary -- a pair killed between
`init` and `step` re-fetches only `step`.

## Filenames are the identity

```
vda_<source>_<backbone>_<height>x<width>_input<inputSize>_infer<inferLen>_<init|step>.tflite
```

The release asset name, the on-disk name, `DepthConfig`'s resolved name and
`RuntimeConfig.modelFileName` are all **the same string**. `<source>` is folded
into the name because release assets are flat; the converter's own output
puts it in a directory instead (`tflite_models/<source>/vda_<backbone>_...`),
so a file copied straight from the converter must be renamed.

`DepthConfig.buildModelFileName()` and `ModelSpec.id` construct this
independently -- keep them byte-identical.

## `source` is NOT the compute device

- `ModelSource` (`gpu`/`original`) = **which file**: which copy of the VDA
  PyTorch source was traced.
- `RuntimeConfig.ComputeDevice` (`CPU`/`GPU`/`NPU`/`AUTO`) = **which delegate**.

An `original` model still contains ops the GPU/NNAPI delegate refuses, and a
delegate is *mandatory* once attached, so the interpreter fails to build. Two
layers guard it, keep both: `DepthViewModel.coerceUnsupportedDevice` forces
CPU up front and toasts (`AUTO` exempt), and `TFLiteModelRunner.loadModel`
catches a rejection and rebuilds on CPU as a backstop.

## "Fully delegated" is not "correct" -- the GPU story

An earlier `gpu` export ran 100% on the GPU delegate, one partition, no
error, ~4x faster than CPU -- and returned a **constant** depth map (every
pixel 0.1804, every frame, any input): a black video. Two independent bugs:

1. The delegate's `MEAN` kernel mis-reduces the `axis=[0, 2]` pattern
   `nn.LayerNorm` lowers to. **Fixed in the converter's `gpu` source**
   (explicit single-axis reduction). Only the re-exported models carry it.
2. The delegate's default **FP16** produces NaN in the motion modules.
   **Fixed in this app**: `TFLiteModelRunner.newGpuDelegate()` sets
   `setPrecisionLossAllowed(false)`. No `.tflite` can enforce this.

Both are required. The symptoms to recognise: a depth video only a few KB for
hundreds of frames; `Done in` with a black preview; identical output for every
frame. The proof of correctness is numeric, not "it ran" -- see
`vda-app-verify` for the CPU-vs-GPU dump recipe that caught it.

Since the `gpu` rewrites are numerically equivalent, a `gpu` pair is never
worse than an `original` one at any device.

## Adding or updating a model

1. Upload **both** halves to the release, following the filename convention.
2. Add a `ModelSpec` to `ModelManifest.ALL` with both **exact byte sizes**.
   The release URL redirects once; read the size from the final hop:
   ```bash
   curl -sIL <url> | grep -i content-length | tail -1
   ```
   Order in `ALL` is the Models page's render order.
3. If it should be fetched on first launch, put it in `BOOTSTRAP`. Downloads
   are serial and every pair is ~240 MB, so bootstrap fetches one.
4. If the backbone is new, `DepthConfig.Variant` needs an entry with the
   matching `backbone` string and the **eight** cache channel widths, read
   from the step model's signature (`logSignature()` or the converter's
   "resolved cache shapes" log). An empty `channels` array makes
   `DepthConfig` throw on construction -- that is intentional.
5. If the resolution, input size or window is new, nothing else changes:
   `DepthConfig` derives the working resolution and `DepthHiddenStates` the
   token counts. Verify them against `logSignature()` once.

> **The manifest's byte sizes are coupled to the release.** No checksums are
> published, so length is the only integrity check. **Re-uploading an asset
> changes its size and makes every installed app reject the new file as a size
> mismatch** -- pair any re-upload with a manifest update in the same change.

## A network note

One Wi-Fi network resolved `release-assets.githubusercontent.com` to a GitHub
Pages address that never answered, so downloads timed out while `github.com`
itself was fine. The app reports it cleanly and retries on tap; there is
nothing to fix in code. If a download "always fails", test another network
before debugging the downloader.
