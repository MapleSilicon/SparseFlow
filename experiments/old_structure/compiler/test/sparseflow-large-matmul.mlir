module {
  func.func @large_matmul() -> tensor<64x64xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<64x64xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<64x64xf32>
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.matmul ins(%cst, %cst_0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }
}
