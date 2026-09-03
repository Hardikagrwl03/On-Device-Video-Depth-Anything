#!/usr/bin/env bash
# Render a .tflite model's op graph to an image, by driving headless Chrome
# to trigger netron's own "Export as SVG"/"Export as PNG" action (not a
# screenshot -- see analysis/visualize.py's docstring for why that
# distinction matters). Thin wrapper around analysis/visualize.py:
# activates the vda-convert conda env and runs from the project root, so
# --tflite/--output paths can be given relative to the project root
# regardless of where you call this from.
#
# Usage:   ./scripts/visualize.sh --tflite <path/to/model.tflite> [visualize.py options]
# Example: ./scripts/visualize.sh --tflite tflite_models/gpu/vda_vits_720x1280_input518_infer8_init.tflite
#
# Defaults to SVG (vector, and the only sane choice for these graphs --
# they commonly render 60000+ px tall); pass --format png for a raster
# image instead. Output defaults to analysis/<source>/<model-basename>.
# <format>, where <source> (original/gpu) is inferred from a *directory*
# component of --tflite's path (tflite_models/<source>/...), same
# convention convert.py/verify.py use -- pass --output explicitly to
# override.
#
# PNG export is noticeably slower than SVG for large graphs (rendered
# through netron's own JS canvas encoder, not a native screenshot) -- a
# few seconds to tens of seconds depending on model size and machine load.
#
# Full option list:
#          ./scripts/visualize.sh --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vda-convert

python analysis/visualize.py "$@"
