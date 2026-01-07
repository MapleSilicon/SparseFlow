module {
  func.func @test_2d(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %init = arith.constant dense<0.0> : tensor<4x4xf32>
    
    %C = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %D = arith.addf %C, %C : tensor<4x4xf32>
    %E = arith.mulf %D, %C : tensor<4x4xf32>
    
    return %E : tensor<4x4xf32>
  }
}
