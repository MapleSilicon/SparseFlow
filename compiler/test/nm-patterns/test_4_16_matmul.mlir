// RUN: mlir-opt %s --sparseflow-spa --sparseflow-rewrite-matmul | FileCheck %s

// Test 4:16 sparsity pattern (25% density)
func.func @test_4_16(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = tensor.empty() : tensor<64x64xf32>
  
  // Annotate with 4:16 pattern
  %1 = "sparseflow.mark"(%arg0) {n = 4 : i32, m = 16 : i32, direction = "row"} : (tensor<64x64xf32>) -> tensor<64x64xf32>
  
  %2 = linalg.matmul 
    ins(%1, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  
  // CHECK: call @sparse_matmul_4_16
  // CHECK-SAME: sparseflow.nm_pattern = "4:16"
  
  return %2 : tensor<64x64xf32>
}
