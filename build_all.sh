#!/bin/bash
set -e

echo "=== SparseFlow v0.1 — Full Pipeline Build & Test ==="

echo ""
echo "=== Cleaning old build directories ==="
rm -rf compiler/build runtime/build

# Choose which MLIR file to use for metadata export
# Default: 64x64 "large" matmul
MLIR_FILE="${SPARSEFLOW_MLIR_FILE:-compiler/test/sparseflow-large-matmul.mlir}"

echo ""
echo "Using MLIR file for metadata export:"
echo "  $MLIR_FILE"

echo ""
echo "=== Step 1: Build SparseFlow Compiler Passes ==="
cd compiler
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm
make -j4 SparseFlowPasses

echo ""
echo "=== Step 2: Generate Hardware Configuration (hardware_config.json) ==="
# CORRECT PATH: Use the correct relative path from compiler/build
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  -pass-pipeline="builtin.module(func.func(sparseflow-export-metadata))" \
  "../../$MLIR_FILE" > /dev/null

echo "Exporting matmul configuration from:"
echo "  $MLIR_FILE"
echo "hardware_config.json written to:"
echo "  $(pwd)/hardware_config.json"

echo ""
echo "=== Step 3: Build Runtime ==="
cd ../../runtime
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j4 sparseflow_test

echo ""
echo "=== Step 4: Run Runtime Demo ==="
./sparseflow_test

echo ""
echo "=== SparseFlow v0.1 Pipeline Complete! ==="