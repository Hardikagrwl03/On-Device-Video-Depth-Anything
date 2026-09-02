#!/usr/bin/env bash
# Benchmark a .tflite model on an Android device's CPU or GPU. Thin wrapper
# around benchmark/benchmark_cpu.sh and benchmark/benchmark_gpu.sh: picks the
# right one for you and runs from the project root, so the model path can be
# given relative to the project root regardless of where you call this from.
#
# Usage:   ./scripts/benchmark.sh <cpu|gpu> <model.tflite> [device_name_or_id]
# Example: ./scripts/benchmark.sh gpu tflite_models/gpu/vda_vits_720x1280_input518_infer8_init.tflite
#
# device_name_or_id is optional if exactly one device is connected via adb.
# Results are logged under benchmark/<tflite_models subfolder>/<cpu|gpu>/.
#
# A converted model is always an init/step *pair* -- benchmark each half
# separately, they are different graphs with different shapes.
#
# Reading a GPU log: the goal state is
#   "INFO: Explicitly applied GPU delegate, and the model graph will be
#    completely executed by the delegate."
# which a --source gpu build should reach. Unsupported ops are reported in
# two wordings, and the second contains neither "ERROR" nor "not supported":
#   "<OP>: Operation is not supported."
#   "<OP>: OP is supported, but tensor type/shape isn't compatible."
# so grep the whole block rather than just ERROR lines:
#   sed -n '/not supported by GPU delegate/,/operations will run/p' <log>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND="${1:-}"

if [ "$BACKEND" = "-h" ] || [ "$BACKEND" = "--help" ]; then
    echo "Usage: $0 <cpu|gpu> <model.tflite> [device_name_or_id]"
    echo "  Benchmarks a .tflite model on an Android device via adb + the TFLite benchmark_model binary."
    echo "  device_name_or_id is optional if exactly one device is connected."
    echo "  Models come as an init/step pair -- benchmark each half separately."
    echo "  Logs land under benchmark/<source>/<cpu|gpu>/."
    exit 0
fi

if [ "$BACKEND" != "cpu" ] && [ "$BACKEND" != "gpu" ]; then
    echo "Usage: $0 <cpu|gpu> <model.tflite> [device_name_or_id]" >&2
    exit 1
fi
shift

cd "$PROJECT_ROOT"
exec "./benchmark/benchmark_${BACKEND}.sh" "$@"
