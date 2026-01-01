module {
  func.func @multiple_matmul(
    %A: tensor<8x8xf32>, %B: tensor<8x8xf32>, 
    %C: tensor<8x8xf32>, %D: tensor<8x8xf32>
  ) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
    %E = tensor.empty() : tensor<8x8xf32>
    %F = tensor.empty() : tensor<8x8xf32>
    
    %R1 = linalg.matmul ins(%A, %B: tensor<8x8xf32>, tensor<8x8xf32>)
                        outs(%E: tensor<8x8xf32>) -> tensor<8x8xf32>
    %R2 = linalg.matmul ins(%C, %D: tensor<8x8xf32>, tensor<8x8xf32>)
                        outs(%F: tensor<8x8xf32>) -> tensor<8x8xf32>
    
    return %R1, %R2 : tensor<8x8xf32>, tensor<8x8xf32>
  }
}
