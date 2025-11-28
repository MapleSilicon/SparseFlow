#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/src/SparseFlow"
RUNTIME_BUILD="$ROOT/runtime/build"
COMPILER_BUILD="$ROOT/compiler/build"

echo "==============================================="
echo "SparseFlow – matrix size + sparsity sweep"
echo "==============================================="
echo

run_case() {
  local label="$1"
  local json="$2"

  echo ">>> $label"
  cp "$COMPILER_BUILD/$json" "$RUNTIME_BUILD/hardware_config.json"
  (
    cd "$RUNTIME_BUILD"
    ./sparseflow_test | grep -E "Matrix:|Total MACs:|Executed MACs:|Theoretical Speedup:"
  )
  echo
}

run_case "64x64x64 (large_matmul)" "hardware_config.json"
run_case "32x32x16 (medium_matmul)" "32x32_config.json"
run_case "16x8x4 (small_matmul)"    "16x8_config.json"