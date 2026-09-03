#!/usr/bin/env bash
# Runs the TFLite benchmark_model binary on an Android device's CPU.
# Usage: ./benchmark_cpu.sh <model.tflite> [device_name_or_id]

set -euo pipefail

MODEL="${1:?Usage: $0 <model.tflite> [device_name_or_id]}"
DEVICE="${2:-}"

if [ ! -f "$MODEL" ]; then
    echo "Model file not found: $MODEL" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="$(basename "$MODEL")"
REMOTE_DIR="/data/local/tmp/vda_benchmark"

ADB_ARGS=()
[ -n "$DEVICE" ] && ADB_ARGS=(-s "$DEVICE")

# Pick the binary matching the device's ABI -- the prebuilt benchmark_model
# binary is architecture-specific and a mismatched one just fails to exec.
ABI="$(adb "${ADB_ARGS[@]}" shell getprop ro.product.cpu.abi | tr -d '\r\n')"
case "$ABI" in
    arm64-v8a)
        BINARY="$SCRIPT_DIR/binary/android_aarch64_benchmark_model"
        ;;
    armeabi-v7a | armeabi)
        BINARY="$SCRIPT_DIR/binary/android_arm_benchmark_model"
        ;;
    *)
        echo "Unsupported device architecture: '$ABI'" >&2
        exit 1
        ;;
esac

adb "${ADB_ARGS[@]}" shell mkdir -p "$REMOTE_DIR"
adb "${ADB_ARGS[@]}" push "$BINARY" "$REMOTE_DIR/benchmark_model" >/dev/null
adb "${ADB_ARGS[@]}" push "$MODEL" "$REMOTE_DIR/$MODEL_NAME" >/dev/null
adb "${ADB_ARGS[@]}" shell chmod +x "$REMOTE_DIR/benchmark_model"

DEVICE_LABEL="${DEVICE:-default}"
DEVICE_LABEL="${DEVICE_LABEL//[:\/]/_}"

# Mirror the model's subdirectory under tflite_models/ (e.g. "original",
# "gpu") as the outer layer under benchmark/, with cpu/ as the inner layer:
# tflite_models/original/foo.tflite's benchmark lands in
# benchmark/original/cpu/. Models outside tflite_models/ land flat in
# benchmark/cpu/.
TFLITE_MODELS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/tflite_models"
MODEL_ABS="$(cd "$(dirname "$MODEL")" && pwd)/$MODEL_NAME"
REL_SUBDIR=""
if [[ "$MODEL_ABS" == "$TFLITE_MODELS_DIR"/* ]]; then
    REL_SUBDIR="$(dirname "${MODEL_ABS#"$TFLITE_MODELS_DIR"/}")"
    [ "$REL_SUBDIR" = "." ] && REL_SUBDIR=""
fi

OUT_DIR="$SCRIPT_DIR${REL_SUBDIR:+/$REL_SUBDIR}/cpu"
LOG="$OUT_DIR/${MODEL_NAME%.tflite}_${DEVICE_LABEL}.log"
mkdir -p "$OUT_DIR"

adb "${ADB_ARGS[@]}" shell "$REMOTE_DIR/benchmark_model" \
    --graph="$REMOTE_DIR/$MODEL_NAME" \
    --num_runs=10 \
    --enable_op_profiling=true \
    --verbose=true 2>&1 | tee "$LOG"
