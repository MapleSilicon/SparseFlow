#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="$HOME/src/SparseFlow/compiler/build"
MLIR_FILE="$HOME/src/SparseFlow/compiler/test/sparseflow-large-matmul.mlir"
OUT_JSON="$HOME/src/SparseFlow/hardware_config.json"

echo "=== SparseFlow: Export metadata from MLIR ==="
echo "MLIR:  $MLIR_FILE"
echo "JSON:  $OUT_JSON"
echo

cd "$BUILD_DIR"

if [ ! -f "$MLIR_FILE" ]; then
  echo "✗ ERROR: MLIR file not found:"
  echo "  $MLIR_FILE"
  exit 1
fi

mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm,sparseflow-export-metadata))' \
  "$MLIR_FILE" > "$OUT_JSON"

echo "✓ hardware_config.json generated"
echo "---- Preview ----"
sed -n '1,40p' "$OUT_JSON" || true
echo "-----------------"
