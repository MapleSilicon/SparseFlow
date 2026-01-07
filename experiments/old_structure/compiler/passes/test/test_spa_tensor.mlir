module {
  func.func @matmul_tensor(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %empty = tensor.empty() : tensor<4x4xf32>
    %result = linalg.matmul 
      ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
