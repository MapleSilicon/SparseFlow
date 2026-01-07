#!/usr/bin/env bash
set -e

# SparseFlow MLIR driver:
#   - Loads the SparseFlow SPA pass plugin
#   - Runs SPA + SPA export on the given MLIR module
#
# Usage:
#   ./sparseflow-opt.sh path/to/model.mlir > /tmp/out.mlir

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

PLUGIN="$ROOT_DIR/compiler/build/passes/SparseFlowPasses.so"

if [ ! -f "$PLUGIN" ]; then
  echo "[sparseflow-opt] ERROR: SparseFlowPasses.so not found at: $PLUGIN" >&2
  echo "[sparseflow-opt] Hint: run ./spa-runner.sh once to build everything." >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo "[sparseflow-opt] Usage: $0 <input.mlir> [extra mlir-opt args...]" >&2
  exit 1
fi

INPUT="$1"
shift

echo "[sparseflow-opt] Running SPA + SPA export on: $INPUT"

mlir-opt-19 \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  -pass-pipeline='builtin.module(sparseflow-spa,sparseflow-spa-export)' \
  "$INPUT" "$@"
