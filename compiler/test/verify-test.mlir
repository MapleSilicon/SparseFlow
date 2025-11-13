// RUN: mlir-opt-19 -load-pass-plugin %llvmshlibdir/SparseFlowPasses%shlibext \
// RUN:   -pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm{n=4 m=2},sparseflow-verify-pattern))' \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID

module {
  func.func @invalid_pattern(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK-INVALID: error: N must be <= M in N:M sparsity, got 4:2
