---
name: vda-setup
description: First-time setup of this Video Depth Anything TFLite converter -- conda env, checkpoints, Android benchmarking prerequisites. Use when asked to set up, install, or get this repo running from a fresh clone, or when convert.py/verify.py fails with a missing-dependency or missing-checkpoint error.
---

# Setting up the VDA converter

## 1. Conda environment

```bash
conda env create -n vda-convert -f environment.yaml
conda activate vda-convert
```

`environment.yaml` is a portable export of the exact environment this
toolkit was built and tested against (Python 3.11, `torch==2.12.0`,
`litert-torch==0.9.3`, `tensorflow`, `decord`, `einops`, etc.). Every script
under `scripts/` activates `vda-convert` for you, so once the env exists
under that exact name you never need to activate it by hand.

`requirements.txt` is the pip equivalent, if you'd rather not use conda:

```bash
pip install -r requirements.txt
```

## 2. Checkpoints

`Video-Depth-Anything/checkpoints/` is gitignored and empty on a fresh
clone. Download the official checkpoints (see
`Video-Depth-Anything/get_weights.sh` for the exact URLs) and name them
`video_depth_anything_<variant>.pth` (`vits`, `vitb`, `vitl`) --
`convert.py`'s default checkpoint lookup depends on this exact naming.

## 3. Android device (optional, only needed for benchmarking)

`benchmark/binary/` already ships prebuilt `benchmark_model` binaries for
both common ABIs, so no build step is needed here. You just need `adb` on
`PATH` and a reachable device:

```bash
adb devices -l
```

`benchmark/benchmark_cpu.sh`/`benchmark_gpu.sh` auto-detect the connected
device's ABI (`arm64-v8a` vs `armeabi-v7a`/`armeabi`) and push the matching
binary. A device on neither ABI (rare -- some x86 emulators) gets an
explicit "Unsupported device architecture" error.

## Verifying the setup worked

```bash
./scripts/convert.sh --variant vits --skip-verify
./scripts/verify.sh --tflite tflite_models/original/video_depth_vits_720x1280_init.tflite
```

If both succeed, the environment and checkpoints are correctly set up. See
the `vda-convert` and `vda-verify` skills for what each does in depth.
