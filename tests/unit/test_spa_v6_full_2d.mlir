module {
  func.func @test_full_2d(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %A_sparse = "sparseflow.mark"(%A)
      {sparseflow.spa_rowmask = [true, false, true, false]}
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %B_sparse = "sparseflow.mark"(%B)
      {sparseflow.spa_colmask = [true, true, false, false]}
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %init = arith.constant dense<0.0> : tensor<4x4xf32>
    
    %C = linalg.matmul ins(%A_sparse, %B_sparse : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    return %C : tensor<4x4xf32>
  }
}
