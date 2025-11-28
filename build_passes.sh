#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/src/SparseFlow/compiler"
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir

cmake --build . --target SparseFlowPasses -j"$(nproc)"
ls -lh passes/SparseFlowPasses.so
