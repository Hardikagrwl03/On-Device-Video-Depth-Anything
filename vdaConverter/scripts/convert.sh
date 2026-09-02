#!/usr/bin/env bash
# Convert a Video Depth Anything checkpoint to TFLite. Thin wrapper around
# convert.py: activates the vda-convert conda env and runs from the project
# root, so paths work regardless of where you call this from.
#
# Usage:   ./scripts/convert.sh [convert.py options]
#
# Example (the GPU-delegate-compatible source -- what you want for on-device
# use; an --source original build falls back heavily to CPU):
#          ./scripts/convert.sh --variant vits --source gpu
#
# Writes an init/step .tflite pair per variant, named
#   tflite_models/<source>/vda_<variant>_<h>x<w>_input<input_size>_infer<infer_len>_<init|step>.tflite
#
# Full option list (variant, source, resolution, input-size, infer-len,
# output paths, ...):
#          ./scripts/convert.sh --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vda-convert

python convert.py "$@"
