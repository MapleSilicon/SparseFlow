#!/bin/bash

set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PLUGIN="compiler/build/passes/SparseFlowPasses.so"
TEST_DIR="compiler/tests"
PASS_PIPELINE="builtin.module(sparseflow-rewrite-matmul)"

if [ ! -f "$PLUGIN" ]; then
  echo -e "${RED}❌ Plugin not found:${NC} $PLUGIN"
  exit 1
fi

patterns=("1_4" "2_4" "2_8" "4_16" "8_32")

overall_rc=0

echo "══════════════════════════════════════════════════════════════"
echo "SparseFlow N:M Rewrite MLIR Tests"
echo "Plugin: $PLUGIN"
echo "Pass pipeline: $PASS_PIPELINE"
echo "══════════════════════════════════════════════════════════════"
echo ""

for p in "${patterns[@]}"; do
  file="${TEST_DIR}/test_nm_rewrite_${p}.mlir"
  echo "▶ Testing pattern ${p} (${file})"
  if [ ! -f "$file" ]; then
    echo -e "  ${RED}❌ Missing test file${NC}"
    overall_rc=1
    continue
  fi

  # Run mlir-opt and show output or error
  mlir-opt-19 \
    --load-pass-plugin="$PLUGIN" \
    --pass-pipeline="$PASS_PIPELINE" \
    "$file" > /tmp/sf_rewrite_${p}.mlir 2> /tmp/sf_rewrite_${p}.err

  rc=$?
  if [ $rc -ne 0 ]; then
    echo -e "  ${RED}❌ FAILED (exit code $rc)${NC}"
    echo "  ── stderr ─────────────────────────────────────────────"
    sed 's/^/    /' /tmp/sf_rewrite_${p}.err
    echo "  ──────────────────────────────────────────────────────"
    overall_rc=1
  else
    echo -e "  ${GREEN}✅ PASSED${NC}"
    # Check that call to sparse_matmul_N_M exists
    fname="sparse_matmul_${p}"
    if grep -q "$fname" /tmp/sf_rewrite_${p}.mlir; then
      echo "  → Found call to ${fname}"
    else
      echo -e "  ${YELLOW}⚠️ No call to ${fname} found in rewritten IR${NC}"
      overall_rc=1
    fi
  fi

  echo ""
done

if [ $overall_rc -eq 0 ]; then
  echo "══════════════════════════════════════════════════════════════"
  echo -e "${GREEN}✅ All N:M rewrite tests PASSED${NC}"
  echo "══════════════════════════════════════════════════════════════"
else
  echo "══════════════════════════════════════════════════════════════"
  echo -e "${RED}❌ Some N:M rewrite tests FAILED${NC}"
  echo "  Check /tmp/sf_rewrite_*.err for details."
  echo "══════════════════════════════════════════════════════════════"
fi

exit $overall_rc
