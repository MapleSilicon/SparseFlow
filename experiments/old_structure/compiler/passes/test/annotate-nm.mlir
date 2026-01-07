// RUN: mlir-opt -load-pass-plugin %shlib -pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm{n=2 m=4}))' %s | FileCheck %s

module {
  func.func @f(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK: linalg.matmul {sparseflow.nm = {m = 4 : i64, n = 2 : i64}}
