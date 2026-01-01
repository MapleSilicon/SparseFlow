#!/usr/bin/env bash
set -euo pipefail

echo "SparseFlow Phase-2B — Build (Runtime Only)"
echo "==========================================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/../build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_BUILD_TYPE=Release ~/CMakeLists.txt
cmake --build . -j"$(nproc)"

echo ""
echo "Build complete ✅"
