#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$HOME/src/SparseFlow"
COMPILER_DIR="$ROOT_DIR/compiler"
RUNTIME_DIR="$ROOT_DIR/runtime"

echo "=== SparseFlow Full Build & Test ==="
echo "ROOT_DIR: $ROOT_DIR"
echo

########################################
# 1. Build compiler passes
########################################
echo "=== [1/3] Building MLIR passes (SparseFlowPasses.so) ==="
cd "$COMPILER_DIR"

mkdir -p build
cd build

echo "--- Running CMake for compiler ---"
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir

echo "--- Running Ninja for SparseFlowPasses ---"
ninja SparseFlowPasses

echo
echo ">>> Built plugin at: $(realpath passes/SparseFlowPasses.so)"
ls -lh passes/SparseFlowPasses.so || true
echo

########################################
# 2. Build runtime + tests
########################################
echo "=== [2/3] Building runtime (sparseflow_test) ==="
cd "$RUNTIME_DIR"

mkdir -p build
cd build

echo "--- Running CMake for runtime ---"
cmake -G Ninja ..

echo "--- Running Ninja for sparseflow_test ---"
ninja sparseflow_test

echo
echo ">>> Built runtime binary at: $(realpath sparseflow_test)"
ls -lh sparseflow_test || true
echo

########################################
# 3. Run runtime test
########################################
echo "=== [3/3] Running SparseFlow v0.1 Runtime Test ==="
./sparseflow_test

echo
echo "=== Done ==="
