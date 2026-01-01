#!/usr/bin/env bash
set -e

# Root of SparseFlow
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 1) Rebuilding SparseFlowPasses.so ==="
cd "$ROOT_DIR/compiler/build"

rm -f passes/SparseFlowPasses.so
make -j4

echo
echo "=== 2) Checking registered SparseFlow passes ==="
mlir-opt-19 --load-pass-plugin=passes/SparseFlowPasses.so --help 2>&1 | grep sparseflow || true

echo
echo "=== 3) Running SPA + SPA Export on test_spa_v6_full_2d.mlir ==="
cd "$ROOT_DIR"

mlir-opt-19 --allow-unregistered-dialect \
  --load-pass-plugin=compiler/build/passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(sparseflow-spa,sparseflow-spa-export)' \
  test_spa_v6_full_2d.mlir

echo
echo "=== 4) spa_sparsity.json (exported sparsity info) ==="
if [ -f spa_sparsity.json ]; then
  cat spa_sparsity.json
else
  echo "❌ spa_sparsity.json not found (export pass did not run or wrote elsewhere)"
fi
