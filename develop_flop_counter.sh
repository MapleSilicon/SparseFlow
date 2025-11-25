#!/bin/bash
cd ~/src/SparseFlow/compiler

echo "=== Building and testing FLOP counter ==="

# Build
cd build
if ninja SparseFlowPasses; then
    echo "✓ Build successful"
    echo "=== Testing FLOP counter ==="
    mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
      --pass-pipeline='builtin.module(func.func(sparseflow-flop-counter))' \
      ../test/sparseflow-large-matmul.mlir 2>&1
else
    echo "✗ Build failed"
    echo "Recent changes in FlopCounterPass.cpp:"
    cd ..
    git diff compiler/passes/FlopCounterPass.cpp | head -30
fi
