#!/usr/bin/env bash
set -euo pipefail

# --- Config (tight, explicit) ---
BUILD_DIR="${BUILD_DIR:-compiler/build}"
PLUGIN_REL="passes/SparseFlowPasses.so"
MLIR_OPT="${MLIR_OPT:-mlir-opt-19}"

# Cost threshold: v0.7 tiled should show true reuse.
MIN_REUSE_X="${MIN_REUSE_X:-32}"

# Golden + negative tests
GOLDEN_MLIR="${GOLDEN_MLIR:-compiler/tests/test_gpu_full_pipeline.mlir}"
NEG_DIR="${NEG_DIR:-compiler/tests/negative}"

echo "=== SparseFlow v0.7 CI Verify ==="
echo "BUILD_DIR=$BUILD_DIR"
echo "MLIR_OPT=$MLIR_OPT"
echo "MIN_REUSE_X=${MIN_REUSE_X}x"
echo "GOLDEN_MLIR=$GOLDEN_MLIR"
echo "NEG_DIR=$NEG_DIR"
echo

# --- Helpers ---
die() { echo "❌ $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || die "Missing file: $1"; }

run_mlir() {
  local pipeline="$1"
  local input="$2"
  "$MLIR_OPT" \
    --load-pass-plugin="$BUILD_DIR/$PLUGIN_REL" \
    --pass-pipeline="$pipeline" \
    "$input"
}

expect_pass() {
  local name="$1"; shift
  echo "✅ EXPECT PASS: $name"
  if ! "$@" >/dev/null 2>&1; then
    echo "----- stderr/stdout -----"
    "$@" 2>&1 | sed -n '1,120p'
    echo "-------------------------"
    die "Expected PASS but failed: $name"
  fi
}

expect_fail_contains() {
  local name="$1"
  local expected_substr="$2"
  shift 2
  echo "✅ EXPECT FAIL: $name"
  set +e
  local out
  out="$("$@" 2>&1)"
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "$out" | sed -n '1,160p'
    die "Expected FAIL but command succeeded: $name"
  fi
  if ! echo "$out" | grep -Fq "$expected_substr"; then
    echo "----- output -----"
    echo "$out" | sed -n '1,200p'
    echo "------------------"
    die "Fail reason mismatch for: $name (missing: $expected_substr)"
  fi
}

parse_reuse_factor_x() {
  awk '
    /reuse factor:/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /x$/) { gsub(/x/, "", $i); print $i; exit }
      }
    }
  '
}

# --- Preconditions ---
need_file "$BUILD_DIR/$PLUGIN_REL"
need_file "$GOLDEN_MLIR"
[[ -d "$NEG_DIR" ]] || die "Missing negative test dir: $NEG_DIR"

# --- 1) Golden: verifier must pass ---
expect_pass "v0.7 verifier passes on golden" \
  run_mlir "builtin.module(sparseflow-spa,sparseflow-rewrite-matmul,sparseflow-gpu-rewrite,gpu.module(sparseflow-gpu-verify-tiled))" \
  "$GOLDEN_MLIR"

# --- 2) Golden: cost model must meet reuse threshold ---
echo "✅ COST MODEL: parsing reuse factor from output"
cost_out="$(run_mlir "builtin.module(sparseflow-spa,sparseflow-rewrite-matmul,sparseflow-gpu-rewrite,gpu.module(sparseflow-gpu-cost-model))" "$GOLDEN_MLIR" 2>&1 || true)"

reuse="$(echo "$cost_out" | parse_reuse_factor_x || true)"
[[ -n "$reuse" ]] || die "Could not parse reuse factor. Output was:\n$(echo "$cost_out" | sed -n '1,200p')"

echo "   reuse factor parsed: ${reuse}x"
if (( reuse < MIN_REUSE_X )); then
  echo "----- cost model output -----"
  echo "$cost_out" | sed -n '1,220p'
  echo "----------------------------"
  die "Reuse factor regression: got ${reuse}x, require >= ${MIN_REUSE_X}x"
fi
echo "✅ COST THRESHOLD OK: ${reuse}x >= ${MIN_REUSE_X}x"
echo

# --- 3) Negative tests: MUST fail for the right reasons ---
need_file "$NEG_DIR/test_missing_barrier.mlir"
need_file "$NEG_DIR/test_missing_shared_memory.mlir"
need_file "$NEG_DIR/test_global_load_in_compute.mlir"

expect_fail_contains "missing barrier fails verifier" \
  "expected >= 2 gpu.barrier" \
  run_mlir "gpu.module(sparseflow-gpu-verify-tiled)" \
  "$NEG_DIR/test_missing_barrier.mlir"

expect_fail_contains "missing shared memory fails verifier" \
  "missing memref.alloca addrspace" \
  run_mlir "gpu.module(sparseflow-gpu-verify-tiled)" \
  "$NEG_DIR/test_missing_shared_memory.mlir"

expect_fail_contains "global load in compute fails verifier" \
  "global loads in compute loop" \
  run_mlir "gpu.module(sparseflow-gpu-verify-tiled)" \
  "$NEG_DIR/test_global_load_in_compute.mlir"

echo
echo "🎉 CI VERIFY PASSED: v0.7 guardrails locked"
