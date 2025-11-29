#!/bin/bash

set -e

echo "=== SparseFlow v0.1 Simple Build ==="

echo ""
echo "=== Step 1: Build Compiler Passes ==="
cd compiler/build
make -j4

echo ""
echo "=== Step 2: Generate Hardware Config ==="
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm,sparseflow-export-metadata))' \
  ../test/sparseflow-large-matmul.mlir > hardware_config.json
echo "✓ hardware_config.json generated"

echo ""
echo "=== Step 3: Build Minimal Runtime ==="
cd ../../runtime/build
make -j4

echo ""
echo "=== Step 4: Run Test ==="
if [ -f "sparseflow_test" ]; then
    echo "✓ Runtime built successfully"
    echo ""
    ./sparseflow_test
else
    echo "✗ Runtime build failed"
    exit 1
fi

echo ""
echo "=== SparseFlow v0.1 Complete! ==="
