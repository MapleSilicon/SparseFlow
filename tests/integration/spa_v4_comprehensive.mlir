module {
  func.func @test(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // Mark A with sparse rows
    %A_sparse = "sparseflow.mark"(%A)
      { sparseflow.spa_rowmask = [true, false, false, true] }
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    
    // Matmul
    %init = arith.constant dense<0.0> : tensor<4x4xf32>
    %C = linalg.matmul ins(%A_sparse, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    // Arithmetic ops
    %D = arith.addf %C, %A_sparse : tensor<4x4xf32>
    %E = arith.mulf %D, %C : tensor<4x4xf32>
    %F = arith.subf %E, %zero : tensor<4x4xf32>
    %G = arith.divf %F, %A_sparse : tensor<4x4xf32>
    
    // ReLU (maximum with zero)
    %H = arith.maximumf %G, %zero : tensor<4x4xf32>
    
    return %H : tensor<4x4xf32>
  }
}
