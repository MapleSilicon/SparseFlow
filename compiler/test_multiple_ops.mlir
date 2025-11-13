module {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %2 = tensor.empty() : tensor<4x4xf32>
    %3 = linalg.matmul ins(%arg1, %arg2 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    return %1, %3 : tensor<4x4xf32>, tensor<4x4xf32>
  }
}
