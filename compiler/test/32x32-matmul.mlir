module {
  func.func @medium_matmul() -> tensor<32x32xf32> {
    %cst = arith.constant dense<1.0> : tensor<32x16xf32>
    %cst_0 = arith.constant dense<2.0> : tensor<16x32xf32>
    %0 = tensor.empty() : tensor<32x32xf32>
    %1 = linalg.matmul ins(%cst, %cst_0 : tensor<32x16xf32>, tensor<16x32xf32>) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
