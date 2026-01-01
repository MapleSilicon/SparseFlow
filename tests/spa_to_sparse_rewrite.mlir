// RUN: mlir-opt %s -sparseflow-spa -sparseflow-rewrite-matmul | FileCheck %s

module {
  func.func @test_rewrite(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %init = arith.constant dense<0.0> : tensor<4x4xf32>
    
    // This matmul will get SPA masks attached
    %C = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    // CHECK: sparseflow.rewrite_target = "sparse_matmul"
    
    return %C : tensor<4x4xf32>
  }
}
