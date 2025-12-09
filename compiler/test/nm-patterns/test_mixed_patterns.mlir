// RUN: mlir-opt %s --sparseflow-spa --sparseflow-rewrite-matmul | FileCheck %s

// Test mixed N:M patterns in same function
func.func @test_mixed(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = tensor.empty() : tensor<16x16xf32>
  
  // First matmul: 1:4
  %2 = "sparseflow.mark"(%arg0) {n = 1 : i32, m = 4 : i32} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  %3 = linalg.matmul 
    ins(%2, %arg1 : tensor<16x16xf32>, tensor<16x16xf32>)
    outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  // CHECK: call @sparse_matmul_1_4
  
  // Second matmul: 2:4
  %4 = "sparseflow.mark"(%3) {n = 2 : i32, m = 4 : i32} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  %5 = linalg.matmul 
    ins(%4, %arg1 : tensor<16x16xf32>, tensor<16x16xf32>)
    outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  // CHECK: call @sparse_matmul_2_4
  
  return %5 : tensor<16x16xf32>
}
