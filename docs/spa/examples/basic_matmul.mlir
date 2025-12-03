// Example 1: Basic matmul with N:M sparsity
// Run: mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
//      --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm),sparseflow-spa)' \
//      basic_matmul.mlir

module {
  func.func @basic_matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %init = arith.constant dense<0.0> : tensor<4x4xf32>
    %C = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %C : tensor<4x4xf32>
  }
}
