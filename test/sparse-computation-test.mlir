module {
  func.func @sparse_matmul(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %init = tensor.empty() : tensor<64x64xf32>
    %result = linalg.matmul
      ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }
}
