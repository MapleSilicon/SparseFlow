module {
  func.func @test(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %A_sparse = "sparseflow.mark"(%A) 
      { sparseflow.spa_rowmask = [true, false, false, true] }
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %B_sparse = "sparseflow.mark"(%B)
      { sparseflow.spa_rowmask = [true, true, false, false] }
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %init = arith.constant dense<0.0> : tensor<4x4xf32>
    
    %C = linalg.matmul ins(%A_sparse, %B_sparse : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %D = arith.addf %C, %A_sparse : tensor<4x4xf32>
    %E = arith.mulf %D, %B_sparse : tensor<4x4xf32>
    
    return %E : tensor<4x4xf32>
  }
}
