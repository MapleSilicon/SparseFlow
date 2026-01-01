#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PLUGIN="$ROOT_DIR/compiler/build/passes/SparseFlowPasses.so"
TEST_MLIR="$ROOT_DIR/compiler/tests/test_nm_rewrite_4_16.mlir"
OUT_IR="$ROOT_DIR/compiler/build/test_4_16.rewritten.mlir"

echo "══════════════════════════════════════════════════════════════"
echo "🔍 Single 4:16 Rewrite Sanity Test (using official test file)"
echo "══════════════════════════════════════════════════════════════"
echo ""

echo "Repo root:      $ROOT_DIR"
echo "Plugin:         $PLUGIN"
echo "Test MLIR:      $TEST_MLIR"
echo "Output IR path: $OUT_IR"
echo ""

if [ ! -f "$PLUGIN" ]; then
  echo -e "${RED}❌ Compiler plugin not found:${NC} $PLUGIN"
  exit 1
fi

if [ ! -f "$TEST_MLIR" ]; then
  echo -e "${RED}❌ Test file not found:${NC} $TEST_MLIR"
  exit 1
fi

cd "$ROOT_DIR/compiler/build"

echo "▶ Running rewrite pass on test_nm_rewrite_4_16.mlir..."
mlir-opt-19 \
  --allow-unregistered-dialect \
  --load-pass-plugin=passes/SparseFlowPasses.so \
  --pass-pipeline="builtin.module(sparseflow-rewrite-matmul)" \
  "$TEST_MLIR" > "$OUT_IR"

echo ""
echo "▶ Checking for sparse_matmul_4_16 call..."
if grep -q "call @sparse_matmul_4_16" "$OUT_IR"; then
  echo -e "${GREEN}✅ PASS:${NC} Rewrite produced call @sparse_matmul_4_16"
else
  echo -e "${RED}❌ FAIL:${NC} No call to @sparse_matmul_4_16 found"
  echo "----- Rewritten IR -----"
  cat "$OUT_IR"
  exit 1
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "🎉 Single 4:16 rewrite sanity test PASSED"
echo "══════════════════════════════════════════════════════════════"
