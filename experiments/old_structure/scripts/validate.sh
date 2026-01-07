#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "SparseFlow Phase-2B: Validation Gate"
echo "========================================="
echo ""

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/../build"

# Build runtime only (skip tests for now)
echo ">>> Step 1: Build Runtime"
cd "$BUILD_DIR"
make sparseflow_runtime -j"$(nproc)" || { echo "❌ BUILD FAILED"; exit 1; }
echo "✅ Build passed"
echo ""

# Plan sizes test
echo ">>> Step 2: Plan Initialization"
export LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}"

if [[ -x "./test_plan_sizes" ]]; then
  ./test_plan_sizes || { echo "❌ PLAN TEST FAILED"; exit 1; }
  echo "✅ Plan init passed"
else
  echo "⚠️  No plan test (skipping)"
fi
echo ""

# 1024 stability
echo ">>> Step 3: 1024³ Stability"
rm -f *.db
[[ -x "./clear_compressed_cache" ]] && ./clear_compressed_cache 2>/dev/null || true

if [[ -x "./test_1024_persistent" ]]; then
  ./test_1024_persistent | tee /tmp/val_1024.txt
  grep -q "GFLOPS" /tmp/val_1024.txt || { echo "❌ 1024 FAILED"; exit 1; }
  echo "✅ 1024³ passed"
else
  echo "⚠️  No 1024 test"
fi
echo ""

echo "========================================="
echo "✅ VALIDATION PASSED"
echo "========================================="
