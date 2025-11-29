#!/bin/bash

set -e

echo "=== SparseFlow v0.1 — Full Pipeline Build & Test ==="

echo ""
echo "=== Cleaning old build directories ==="
rm -rf compiler/build runtime/build

echo ""
echo "=== Step 1: Build SparseFlow Compiler Passes ==="
cd compiler
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm
make -j4

echo ""
echo "=== Step 2: Generate Hardware Configuration (hardware_config.json) ==="
if [ -f "../test/sparseflow-large-matmul.mlir" ]; then
    mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
      --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm,sparseflow-export-metadata))' \
      ../test/sparseflow-large-matmul.mlir > hardware_config.json
    echo "✓ hardware_config.json generated"
    echo "File: $(pwd)/hardware_config.json"
else
    echo "✗ sparseflow-large-matmul.mlir not found"
    echo "Available test files:"
    ls ../test/ || echo "No test directory"
    exit 1
fi

echo ""
echo "=== Step 3: Build SparseFlow Runtime ==="
cd ../../runtime
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j4

echo ""
echo "=== Step 4: Test Complete Pipeline ==="
if [ -f "sparseflow_test" ]; then
    echo "✓ Runtime built successfully"
    echo ""
    ./sparseflow_test
else
    echo "✗ Failed to build runtime"
    exit 1
fi

echo ""
echo "=== SparseFlow v0.1 Pipeline Complete! ==="
