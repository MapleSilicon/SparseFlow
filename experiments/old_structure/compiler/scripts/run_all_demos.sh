#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
BUILD_DIR="$ROOT_DIR/build"

echo "======================================="
echo " 🚀 SparseFlow — Full Demo Pipeline"
echo "======================================="

echo ""
echo "🔧 Step 1 — Rebuild SparseFlow"
echo "---------------------------------------"
cd "$ROOT_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir

make -j"$(nproc)"

echo ""
echo "✔ Build completed successfully"

echo ""
echo "======================================="
echo " 🧪 Step 2 — SPA Tensor Demo"
echo "======================================="
cd "$ROOT_DIR"
./scripts/run_spa_tensor_demo.sh || { echo "❌ SPA tensor demo failed"; exit 1; }
echo "✔ SPA tensor demo OK"

echo ""
echo "======================================="
echo " 🔍 Step 3 — FLOP Counter Demo"
echo "======================================="
./scripts/run_flop_counter_demo.sh || { echo "❌ FLOPs demo failed"; exit 1; }
echo "✔ FLOPs demo OK"

echo ""
echo "======================================="
echo " 🎯 Step 4 — SPA + FLOPs Combined Demo"
echo "======================================="
./scripts/run_spa_plus_flops_demo.sh || { echo "❌ SPA + FLOPs demo failed"; exit 1; }
echo "✔ Combined SPA + FLOPs demo OK"

echo ""
echo "======================================="
echo " 📦 Step 5 — SPA JSON Export Demo"
echo "======================================="
./scripts/run_spa_json_export_demo.sh || { echo "❌ JSON export demo failed"; exit 1; }
echo "✔ JSON export demo OK"

echo ""
echo "✅ All SparseFlow demos completed successfully."
