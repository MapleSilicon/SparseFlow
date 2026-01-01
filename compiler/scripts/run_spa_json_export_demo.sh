#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
BUILD_DIR="$SCRIPT_DIR/../build"
JSON_PATH="$SCRIPT_DIR/../spa_sparsity.json"

# Remove old JSON
rm -f "$JSON_PATH"

# Run SPA + export to JSON
mlir-opt-19 \
  -load-pass-plugin "$BUILD_DIR/passes/SparseFlowPasses.so" \
  -pass-pipeline='builtin.module(sparseflow-spa,sparseflow-spa-export)' \
  "$BUILD_DIR/../passes/test/test_spa_tensor.mlir" -o /dev/null

echo "✅ Exported sparsity info to $JSON_PATH"
echo "=== SPA sparsity JSON ==="
cat "$JSON_PATH"
