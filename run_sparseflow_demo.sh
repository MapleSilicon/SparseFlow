#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           SparseFlow Compiler Demo v0.1                       ║"
echo "║  End-to-End: MLIR → SPA → Rewrite → LLVM → JIT → Performance ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Build if needed
if [ ! -f "compiler/build/sparseflow-runner" ]; then
    echo "Building SparseFlow..."
    cd compiler/build
    cmake .. \
      -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
      -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm
    make -j4
    cd ../..
fi

export LD_LIBRARY_PATH=~/src/SparseFlow/runtime/build:$LD_LIBRARY_PATH

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Correctness Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd compiler/build
./test_jit_correctness
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Performance Benchmarks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./benchmark_suite
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Full Compiler Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./run_e2e_pipeline.sh
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Demo Complete!                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
