module {
  func.func @test_sparse_matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>)
      -> tensor<4x4xf32> {

    %c = arith.constant dense<0.0> : tensor<4x4xf32>

    %0 = linalg.matmul
      { "sparseflow.sparsity" = "2:4" }
      ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>

    return %0 : tensor<4x4xf32>
  }
}
