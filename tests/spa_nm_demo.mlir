module {
  func.func @matmul_test(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>)
      -> tensor<4x4xf32> {

    %zero = arith.constant dense<0.0> : tensor<4x4xf32>

    %C = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%zero : tensor<4x4xf32>) -> tensor<4x4xf32>

    %D = arith.addf %C, %C : tensor<4x4xf32>
    return %D : tensor<4x4xf32>
  }
}
