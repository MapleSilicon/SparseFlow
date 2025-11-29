#!/usr/bin/env bash
set -e

# Go to build directory
cd "$(dirname "$0")/../compiler/build"

echo "=== Building SparseFlowPasses ==="
ninja SparseFlowPasses

echo
echo "=== Creating test_annotate.mlir (single 4x4 matmul) ==="
cat > test_annotate.mlir <<'MLIR'
module {
  func.func @simple_matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %C = tensor.empty() : tensor<4x4xf32>
    %0 = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                      outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
MLIR

echo
echo "=== Running SparseFlow pipeline on test_annotate.mlir ==="
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  -pass-pipeline='builtin.module(
    func.func(
      sparseflow-annotate-nm{n=2 m=4},
      sparseflow-flop-counter
    ),
    sparseflow-export-metadata
  )' \
  test_annotate.mlir

echo
echo "=== sparseflow_metadata.json (single matmul) ==="
if [ -f sparseflow_metadata.json ]; then
  cat sparseflow_metadata.json
else
  echo "sparseflow_metadata.json not found!"
fi

echo
echo "=== Creating test_multi_matmul.mlir (two ops: 4x4 + 8x8) ==="
cat > test_multi_matmul.mlir <<'MLIR'
module {
  // First matmul: 4x4 * 4x4
  func.func @small_matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %C = tensor.empty() : tensor<4x4xf32>
    %0 = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                      outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Second matmul: 8x8 * 8x8
  func.func @large_matmul(%X: tensor<8x8xf32>, %Y: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %Z = tensor.empty() : tensor<8x8xf32>
    %1 = linalg.matmul ins(%X, %Y : tensor<8x8xf32>, tensor<8x8xf32>)
                      outs(%Z : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
  }
}
MLIR

echo
echo "=== Running SparseFlow pipeline on test_multi_matmul.mlir ==="
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  -pass-pipeline='builtin.module(
    func.func(
      sparseflow-annotate-nm{n=2 m=4},
      sparseflow-flop-counter
    ),
    sparseflow-export-metadata
  )' \
  test_multi_matmul.mlir

echo
echo "=== sparseflow_metadata.json (multi-matmul) ==="
if [ -f sparseflow_metadata.json ]; then
  cat sparseflow_metadata.json
else
  echo "sparseflow_metadata.json not found!"
fi
