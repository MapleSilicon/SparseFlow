module {
  func.func @f(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %C = tensor.empty() : tensor<4x4xf32>
    %R = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                        outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %R : tensor<4x4xf32>
  }
}
