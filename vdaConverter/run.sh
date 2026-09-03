#!/usr/bin/env bash
# End-to-end pipeline for one variant/source/resolution combination:
#   1. convert.sh   -- trace + export the init/step .tflite pair
#   2. compare.sh   -- pure-PyTorch fidelity check (original code vs both
#                       wrapper sources), independent of the .tflite files
#   3. verify.sh    -- wrapper vs exported .tflite, run once per model file
#                       (init and step) for symmetry with steps 4/5 below,
#                       even though either call alone already checks both
#                       signatures via verify.py's automatic sibling lookup
#   4. benchmark.sh -- on-device latency, once per model file (init, step)
#   5. visualize.sh -- op-graph image, once per model file (init, step)
#
# Usage:
#   ./run.sh [options]
#
# Example (equivalent to just ./run.sh, given the defaults below):
#   ./run.sh --variant vits --source gpu --backend gpu
#
# Options (all optional):
#   --variant {vits,vitb,vitl} Encoder variant to convert (default: vits --
#                              the only variant with a checkpoint present
#                              in Video-Depth-Anything/checkpoints/ by
#                              default; get others via get_weights.sh).
#   --source {original,gpu}   Model tree to trace from (default: gpu, the
#                              TFLite-GPU-delegate-compatible build this
#                              project exists for). original imports the
#                              unmodified upstream code instead.
#   --checkpoint PATH         .pth checkpoint (default: Video-Depth-Anything/
#                              checkpoints/video_depth_anything_<variant>.pth).
#   --height N                 Traced input height (default: 720).
#   --width N                  Traced input width (default: 1280).
#   --input-size N              Base working resolution (default: 518).
#   --infer-len N               Context length baked into the cache (default: 8).
#   --output-dir DIR          Where convert.sh writes the .tflite pair
#                              (default: tflite_models/<source>).
#   --backend {cpu,gpu}       Delegate to benchmark on-device (default: gpu).
#   --device ID_OR_NAME        adb device, if more than one is connected.
#   --format {svg,png}         visualize.sh output format (default: svg).
#
# Each step's own script has a more detailed --help (./scripts/convert.sh
# --help etc.) -- this wrapper only exposes the flags needed to chain them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VARIANT="vits"
SOURCE="gpu"
CHECKPOINT=""
HEIGHT="720"
WIDTH="1280"
INPUT_SIZE="518"
INFER_LEN="8"
OUTPUT_DIR=""
BACKEND="gpu"
DEVICE=""
FORMAT="svg"

usage() {
    sed -n '2,41p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --width) WIDTH="$2"; shift 2 ;;
        --input-size) INPUT_SIZE="$2"; shift 2 ;;
        --infer-len) INFER_LEN="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --format) FORMAT="$2"; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$VARIANT" in
    vits | vitb | vitl) ;;
    *) echo "Invalid --variant: $VARIANT (must be vits, vitb, or vitl)" >&2; exit 1 ;;
esac
case "$SOURCE" in
    original | gpu) ;;
    *) echo "Invalid --source: $SOURCE (must be original or gpu)" >&2; exit 1 ;;
esac
case "$BACKEND" in
    cpu | gpu) ;;
    *) echo "Invalid --backend: $BACKEND (must be cpu or gpu)" >&2; exit 1 ;;
esac

CONVERT_ARGS=(--variant "$VARIANT" --source "$SOURCE" --height "$HEIGHT" --width "$WIDTH" --input-size "$INPUT_SIZE" --infer-len "$INFER_LEN")
COMPARE_ARGS=(--variant "$VARIANT" --height "$HEIGHT" --width "$WIDTH" --input-size "$INPUT_SIZE" --infer-len "$INFER_LEN")
[ -n "$CHECKPOINT" ] && CONVERT_ARGS+=(--checkpoint "$CHECKPOINT") && COMPARE_ARGS+=(--checkpoint "$CHECKPOINT")
[ -n "$OUTPUT_DIR" ] && CONVERT_ARGS+=(--output-dir "$OUTPUT_DIR")

echo "==> [1/5] convert.sh"
CONVERT_LOG="$(mktemp)"
trap 'rm -f "$CONVERT_LOG"' EXIT
./scripts/convert.sh "${CONVERT_ARGS[@]}" 2>&1 | tee "$CONVERT_LOG"

# Read the exported paths back out of convert.py's own "conversion
# successful: <path>" lines rather than recomputing its output-filename
# convention here -- one source of truth, no risk of the two drifting apart.
mapfile -t MODEL_PATHS < <(sed -n 's/^ *conversion successful: //p' "$CONVERT_LOG")
if [ "${#MODEL_PATHS[@]}" -ne 2 ]; then
    echo "Expected convert.sh to report exactly 2 exported models, found ${#MODEL_PATHS[@]}:" >&2
    printf '  %s\n' "${MODEL_PATHS[@]}" >&2
    exit 1
fi
INIT_PATH="${MODEL_PATHS[0]}"
STEP_PATH="${MODEL_PATHS[1]}"
echo "    init: $INIT_PATH"
echo "    step: $STEP_PATH"

echo "==> [2/5] compare.sh"
./scripts/compare.sh "${COMPARE_ARGS[@]}"

for MODEL in "$INIT_PATH" "$STEP_PATH"; do
    echo "==> [3/5] verify.sh $MODEL"
    # --source is passed explicitly rather than left to verify.sh's own
    # path-based inference: that inference only works when --output-dir
    # kept "gpu"/"original" as a path component, and we already know the
    # real answer here regardless of --output-dir. Getting this wrong
    # would silently build the WRONG PyTorch reference to compare
    # against (verify.py falls back to "original") -- a false pass, not
    # an error, since most gpu/ edits are near-exact rewrites anyway.
    ./scripts/verify.sh --tflite "$MODEL" --source "$SOURCE"
done

BENCHMARK_ARGS=("$BACKEND")
for MODEL in "$INIT_PATH" "$STEP_PATH"; do
    echo "==> [4/5] benchmark.sh $BACKEND $MODEL"
    if [ -n "$DEVICE" ]; then
        ./scripts/benchmark.sh "$BACKEND" "$MODEL" "$DEVICE"
    else
        ./scripts/benchmark.sh "$BACKEND" "$MODEL"
    fi
done

for MODEL in "$INIT_PATH" "$STEP_PATH"; do
    echo "==> [5/5] visualize.sh $MODEL"
    # --output is computed explicitly for the same reason --source is
    # passed to verify.sh above: visualize.py's own analysis/<source>/...
    # default also comes from a path-based guess, which only lands right
    # when --output-dir happened to keep "gpu"/"original" as a path
    # component. No correctness impact here (unlike verify.py) since
    # visualize.py builds nothing from source -- just a filing location.
    BASENAME="$(basename "${MODEL%.tflite}")"
    ./scripts/visualize.sh --tflite "$MODEL" --format "$FORMAT" --output "analysis/$SOURCE/$BASENAME.$FORMAT"
done

echo "==> done"
echo "    init: $INIT_PATH"
echo "    step: $STEP_PATH"
