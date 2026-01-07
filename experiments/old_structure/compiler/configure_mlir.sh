#!/usr/bin/env bash
set -e

# Configure SparseFlow compiler against system LLVM/MLIR 19
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -G "Unix Makefiles" \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm
