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
cd "$COMPILER_BUILD"
cmake --build . --target SparseFlowPasses -j4
echo "✅ Compiler passes built"

echo
echo "🔧 Step 2: Running SparseFlow Compiler Pipeline..."
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm,sparseflow-flop-counter,sparseflow-export-metadata))' \
  "$TEST_MLIR" > hardware_config.json 2>&1

echo "✅ FLOPs: $(grep 'Total FLOPs' hardware_config.json)"
echo "✅ Hardware config generated"

echo
echo "⚡ Step 3: Building MapleSilicon Runtime..."
cd "$RUNTIME_BUILD"
cmake --build . --target sparseflow_test -j4
echo "✅ Runtime built"

echo
echo "🚀 Step 4: Executing Complete SparseFlow Pipeline..."
echo "================================================================================"
./sparseflow_test
echo "================================================================================"

echo
echo "🎉 SPARSEFLOW v0.1 VALIDATION COMPLETE!"
