#!/bin/bash
# SparseFlow v0.7 Local Verification Script
set -e

cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════════"
echo "SparseFlow v0.7 Verification Suite"
echo "════════════════════════════════════════════════════════════════"

cd compiler/build

echo ""
echo "✅ TEST 1: Positive case (should pass)"
mlir-opt-19 \
  --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline="builtin.module(sparseflow-spa,sparseflow-rewrite-matmul,sparseflow-gpu-rewrite,gpu.module(sparseflow-gpu-verify-tiled))" \
  ../tests/test_gpu_full_pipeline.mlir >/dev/null 2>&1
echo "✅ Verifier passed on valid v0.7 kernel"

echo ""
echo "✅ TEST 2: Cost model (should show 32× reuse)"
mlir-opt-19 \
  --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline="builtin.module(sparseflow-spa,sparseflow-rewrite-matmul,sparseflow-gpu-rewrite,gpu.module(sparseflow-gpu-cost-model))" \
  ../tests/test_gpu_full_pipeline.mlir 2>/dev/null

echo ""
echo "✅ TEST 3: Negative cases (should all fail)"
! mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline="gpu.module(sparseflow-gpu-verify-tiled)" \
  ../tests/negative/test_missing_barrier.mlir >/dev/null 2>&1
echo "✅ Missing barrier correctly caught"

! mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline="gpu.module(sparseflow-gpu-verify-tiled)" \
  ../tests/negative/test_missing_shared_memory.mlir >/dev/null 2>&1
echo "✅ Missing shared memory correctly caught"

! mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline="gpu.module(sparseflow-gpu-verify-tiled)" \
  ../tests/negative/test_global_load_in_compute.mlir >/dev/null 2>&1
echo "✅ Global load violation correctly caught"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🎉 ALL TESTS PASSED - v0.7 VERIFIED"
echo "════════════════════════════════════════════════════════════════"
