#!/bin/bash
cd ~/src/SparseFlow/compiler/build

echo "=== Building SparseFlow passes ==="
ninja SparseFlowPasses

if [ $? -eq 0 ]; then
    echo "✓ Build successful"
    echo ""
    echo "=== Testing FLOP counter ==="
    mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
      --pass-pipeline='builtin.module(func.func(sparseflow-flop-counter))' \
      ../../test/simple-matmul.mlir 2>&1
else
    echo "✗ Build failed"
    exit 1
fi
