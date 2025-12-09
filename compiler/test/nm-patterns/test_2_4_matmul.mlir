// RUN: mlir-opt %s --sparseflow-spa --sparseflow-rewrite-matmul | FileCheck %s

// Test 2:4 sparsity pattern (50% density)
func.func @test_2_4(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = tensor.empty() : tensor<16x16xf32>
  
  // Annotate with 2:4 pattern
  %1 = "sparseflow.mark"(%arg0) {n = 2 : i32, m = 4 : i32, direction = "row"} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  
  %2 = linalg.matmul 
    ins(%1, %arg1 : tensor<16x16xf32>, tensor<16x16xf32>)
    outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  // CHECK: call @sparse_matmul_2_4
  // CHECK-SAME: sparseflow.nm_pattern = "2:4"
  
  return %2 : tensor<16x16xf32>
}
