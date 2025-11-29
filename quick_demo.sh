#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPILER_BUILD="$ROOT_DIR/compiler/build"
RUNTIME_BUILD="$ROOT_DIR/runtime/build"
MLIR_INPUT="$ROOT_DIR/compiler/test/sparseflow-large-matmul.mlir"
JSON_OUT="$COMPILER_BUILD/hardware_config.json"

echo "=== SparseFlow Quick Demo ==="

# Sanity checks
if [ ! -f "$COMPILER_BUILD/passes/SparseFlowPasses.so" ]; then
  echo "[!] Compiler plugin not found at:"
  echo "    $COMPILER_BUILD/passes/SparseFlowPasses.so"
  echo "    Run ./build_all.sh first."
  exit 1
fi

if [ ! -x "$RUNTIME_BUILD/sparseflow_test" ]; then
  echo "[!] Runtime binary not found at:"
  echo "    $RUNTIME_BUILD/sparseflow_test"
  echo "    Run ./build_all.sh first."
  exit 1
fi

if [ ! -f "$JSON_OUT" ]; then
  echo "[!] $JSON_OUT not found."
  echo "    Run ./build_all.sh first so the compiler pipeline can export it."
  exit 1
fi

echo ""
echo "=== Step 1: Re-using existing hardware_config.json from build_all.sh ==="
echo "MLIR (source): $MLIR_INPUT"
echo "JSON config:   $JSON_OUT"
ls -lh "$JSON_OUT"

echo ""
echo "=== Step 2: Run SparseFlow runtime with that config ==="
cd "$RUNTIME_BUILD"
./sparseflow_test
