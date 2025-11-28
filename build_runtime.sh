#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/src/SparseFlow/runtime"
mkdir -p build
cd build

cmake ..
cmake --build . --target sparseflow_test -j"$(nproc)"
ls -lh sparseflow_test
