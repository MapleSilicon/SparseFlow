module {
  func.func @test_transpose(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // Matrix with row sparsity
    %A_sparse = "sparseflow.mark"(%A)
      {sparseflow.spa_rowmask = [true, false, true, false],
       sparseflow.spa_colmask = [true, true, false, false]}
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %init = arith.constant dense<0.0> : tensor<4x4xf32>
    
    // Transpose should swap rows and cols
    %B = linalg.transpose ins(%A_sparse : tensor<4x4xf32>)
                          outs(%init : tensor<4x4xf32>)
                          permutation = [1, 0]
    
    return %B : tensor<4x4xf32>
  }
}
