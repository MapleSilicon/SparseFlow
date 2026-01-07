#!/usr/bin/env bash
set -euo pipefail

# Hard-coded root because Windows/WSL + dirname was causing pain
ROOT_DIR="/home/maplesilicon/src/SparseFlow"
BUILD_DIR="$ROOT_DIR/compiler/build"

echo "[spa_v06_export] ROOT_DIR = $ROOT_DIR"
echo "[spa_v06_export] BUILD_DIR = $BUILD_DIR"

cd "$BUILD_DIR"

# Sanity check
if [ ! -f passes/SparseFlowPasses.so ]; then
  echo "[spa_v06_export] ERROR: passes/SparseFlowPasses.so not found. Build the passes first."
  exit 1
fi

if [ ! -f "$ROOT_DIR/spa_v6_matmul_2d.mlir" ]; then
  echo "[spa_v06_export] ERROR: $ROOT_DIR/spa_v6_matmul_2d.mlir not found."
  exit 1
fi

echo "[spa_v06_export] Removing old spa_sparsity.json (if any)..."
rm -f spa_sparsity.json

echo "[spa_v06_export] Running pipeline: builtin.module(sparseflow-spa, sparseflow-spa-export)..."

mlir-opt-19 \
  --allow-unregistered-dialect \
  --load-pass-plugin=passes/SparseFlowPasses.so \
  --pass-pipeline="builtin.module(sparseflow-spa, sparseflow-spa-export)" \
  "$ROOT_DIR/spa_v6_matmul_2d.mlir"

echo "[spa_v06_export] Done. Checking spa_sparsity.json..."
python3 -m json.tool spa_sparsity.json | head -80
