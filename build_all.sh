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
make -j4 SparseFlowPasses

echo ""
echo "=== Step 2: Generate Hardware Configuration (hardware_config.json) ==="
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  -pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm),sparseflow-export-metadata)' \
  ../test/sparseflow-large-matmul.mlir \
  -o /dev/null > hardware_config.json

echo "hardware_config.json written to:"
echo "  $(pwd)/hardware_config.json"

echo ""
echo "=== Step 3: Build Runtime ==="
cd ../../runtime
mkdir -p build
cd build
cmake ..
make -j4 sparseflow_test

echo ""
echo "=== Step 4: Run Runtime Demo ==="
./sparseflow_test

echo ""
echo "=== SparseFlow v0.1 Pipeline Complete! ==="
