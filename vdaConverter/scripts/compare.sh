#!/usr/bin/env bash
# Compare the untouched original VDA model against both wrapper sources on
# identical inputs -- pure PyTorch, no TFLite, no device. Thin wrapper around
# compare.py: activates the vda-convert conda env and runs from the project
# root, so --checkpoint paths can be given relative to the project root
# regardless of where you call this from.
#
# Usage:   ./scripts/compare.sh --variant {vits,vitb,vitl} [compare.py options]
# Example: ./scripts/compare.sh --variant vits
#
# This is the tool to run while editing
# Video-Depth-Anything/video_depth_anything_gpu/ -- it is what proves a
# GPU-delegate-compatibility rewrite didn't change the model's behaviour.
# It runs three pipelines (the original model's own infer_video_depth_one(),
# wrapper(source=original), wrapper(source=gpu)) over a two-frame sequence,
# exercising both the init and step paths.
#
# Two tolerances, because the pairs have different expected floors:
#   --atol       (5e-5) wrapper(original) vs wrapper(gpu) -- expect 0.000000
#                       for nearly every edit; a nonzero value is a signal,
#                       not noise. It is 5e-5 rather than 0 only because of
#                       the linear_as_conv1x1 substitution.
#   --code-atol  (1e-3) original code vs either wrapper -- never 0, since the
#                       wrapper resizes/normalizes in torch ops while
#                       upstream uses cv2/numpy.
#
# Passing here is a precondition, not a completion: it says the edit is
# behaviour-preserving, not that it helped the GPU delegate. Confirm that by
# re-converting and running ./scripts/benchmark.sh gpu ...
#
# Full option list:
#          ./scripts/compare.sh --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vda-convert

python compare.py "$@"
