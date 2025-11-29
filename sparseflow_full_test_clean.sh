#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "SPARSEFLOW v0.1 - CLEAN END-TO-END VALIDATION"
echo "================================================================================"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILER_BUILD="$SCRIPT_DIR/compiler/build"
RUNTIME_BUILD="$SCRIPT_DIR/runtime/build"
TEST_MLIR="$SCRIPT_DIR/compiler/test/sparseflow-large-matmul.mlir"

echo "📦 Step 1: Building MLIR Compiler Passes..."
mkdir -p "$COMPILER_BUILD"
cd "$COMPILER_BUILD"
cmake .. >/dev/null 2>&1 || true
cmake --build . --target SparseFlowPasses -j4
echo "✅ Compiler passes built"

echo
echo "🔧 Step 2: Running SparseFlow Compiler Pipeline (export metadata)..."
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(sparseflow-export-metadata)' \
  "$TEST_MLIR" > "$SCRIPT_DIR/hardware_config.json"
echo "✅ hardware_config.json generated at $SCRIPT_DIR/hardware_config.json"

echo
echo "⚡ Step 3: Building MapleSilicon Runtime..."
mkdir -p "$RUNTIME_BUILD"
cd "$RUNTIME_BUILD"
cmake .. >/dev/null 2>&1 || true
cmake --build . --target sparseflow_test -j4
echo "✅ Runtime built"

echo
echo "🚀 Step 4: Executing Sparse Matmul..."
cp "$SCRIPT_DIR/hardware_config.json" .
./sparseflow_test

echo
echo "🎉 SPARSEFLOW v0.1 VALIDATION COMPLETE (clean run)!"
