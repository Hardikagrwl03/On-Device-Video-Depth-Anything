#!/usr/bin/env bash
# Verify an exported .tflite pair against the PyTorch wrapper it was traced
# from (conversion fidelity). Thin wrapper around verify.py: activates the
# vda-convert conda env and runs from the project root, so
# --tflite/--checkpoint paths can be given relative to the project root
# regardless of where you call this from.
#
# Usage:   ./scripts/verify.sh --tflite <path/to/model_init.tflite> [verify.py options]
# Example: ./scripts/verify.sh --tflite tflite_models/gpu/vda_vits_720x1280_input518_infer8_init.tflite
#
# Pass either half of the pair -- the sibling is found automatically, and
# both the init and step signatures are checked.
#
# --variant/--source/--height/--width/--input-size/--infer-len are normally
# left out: they're inferred from the --tflite filename and path. The one
# case that needs care is --source, which is inferred from a *directory*
# component (tflite_models/<source>/...) -- if you converted with a custom
# --output-dir that has neither "gpu" nor "original" as a path segment, pass
# --source explicitly or the wrong wrapper gets built.
#
# This compares wrapper vs .tflite only. To check a video_depth_anything_gpu/
# edit against the untouched original model, use ./scripts/compare.sh.
#
# Full option list:
#          ./scripts/verify.sh --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vda-convert

python verify.py "$@"
