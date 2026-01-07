#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="$(dirname "$0")/../build"

mlir-opt-19 \
  -load-pass-plugin "$BUILD_DIR/passes/SparseFlowPasses.so" \
  -pass-pipeline='builtin.module(sparseflow-spa,func.func(sparseflow-flop-counter))' \
  "$BUILD_DIR/../passes/test/test_spa_tensor.mlir" -o -
