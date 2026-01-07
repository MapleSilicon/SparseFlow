#!/usr/bin/env bash
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "🔬 SparseFlow v0.2 - MLIR Integration Tests"
echo "═══════════════════════════════════════════════════════════════"

PLUGIN="compiler/build/passes/SparseFlowPasses.so"
TEST_DIR="compiler/tests"
PASS_PIPELINE="builtin.module(sparseflow-rewrite-matmul)"

if [ ! -f "$PLUGIN" ]; then
  echo "❌ Missing plugin: $PLUGIN"
  exit 1
fi

echo "Plugin: $PLUGIN"
echo "Tests: $TEST_DIR"
echo ""

FAIL=0
PASS_COUNT=0
FAIL_COUNT=0

for f in "$TEST_DIR"/test_nm_rewrite_*.mlir; do
  [ -e "$f" ] || continue
  TEST_NAME=$(basename "$f")
  echo "Testing: $TEST_NAME"
  
  if mlir-opt-19 \
        --load-pass-plugin="$PLUGIN" \
        --pass-pipeline="$PASS_PIPELINE" \
        "$f" >/dev/null 2>&1; then
    echo "  ✅ PASS"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo "  ❌ FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAIL=1
  fi
done

echo ""
echo "Results: $PASS_COUNT passed, $FAIL_COUNT failed"

if [ "$FAIL" -ne 0 ]; then
  echo "❌ MLIR integration tests FAILED"
  exit 1
fi

echo "✅ All MLIR integration tests PASSED"
