module {
  func.func @small_matmul() -> tensor<16x8xf32> {
    %cst = arith.constant dense<1.0> : tensor<16x4xf32>
    %cst_0 = arith.constant dense<2.0> : tensor<4x8xf32>
    %0 = tensor.empty() : tensor<16x8xf32>
    %1 = linalg.matmul ins(%cst, %cst_0 : tensor<16x4xf32>, tensor<4x8xf32>) outs(%0 : tensor<16x8xf32>) -> tensor<16x8xf32>
    return %1 : tensor<16x8xf32>
  }
}
