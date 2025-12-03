module {
  // Test 1: Matmul with N:M
  func.func @test_matmul_nm(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %init = arith.constant dense<0.0> : tensor<8x8xf32>
    %C = linalg.matmul ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>)
                       outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %C : tensor<8x8xf32>
  }
  
  // Test 2: Arithmetic chain
  func.func @test_arithmetic(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %B = arith.addf %A, %A : tensor<4x4xf32>
    %C = arith.mulf %B, %A : tensor<4x4xf32>
    %D = arith.subf %C, %A : tensor<4x4xf32>
    return %D : tensor<4x4xf32>
  }
  
  // Test 3: ReLU pattern
  func.func @test_relu(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    %relu = arith.maximumf %A, %zero : tensor<4x4xf32>
    return %relu : tensor<4x4xf32>
  }
  
  // Test 4: 2D propagation
  func.func @test_2d(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
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
