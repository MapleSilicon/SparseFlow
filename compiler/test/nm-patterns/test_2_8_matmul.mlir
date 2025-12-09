// RUN: mlir-opt %s --sparseflow-spa --sparseflow-rewrite-matmul | FileCheck %s

// Test 2:8 sparsity pattern (25% density)
func.func @test_2_8(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  
  // Annotate with 2:8 pattern
  %1 = "sparseflow.mark"(%arg0) {n = 2 : i32, m = 8 : i32, direction = "row"} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  
  %2 = linalg.matmul 
    ins(%1, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  
  // CHECK: call @sparse_matmul_2_8
  // CHECK-SAME: sparseflow.nm_pattern = "2:8"
  
  return %2 : tensor<32x32xf32>
}
