module {
  func.func @test(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // A: rows [0,2] are maybe non-zero; others provably zero
    %A_mark = "sparseflow.mark"(%A)
      { sparseflow.spa_rowmask = [true, false, true, false] }
      : (tensor<4x4xf32>) -> tensor<4x4xf32>

    // B: cols [1,3] are maybe non-zero; others provably zero
    %B_mark = "sparseflow.mark"(%B)
      { sparseflow.spa_colmask = [false, true, false, true] }
      : (tensor<4x4xf32>) -> tensor<4x4xf32>

    %init = arith.constant dense<0.0> : tensor<4x4xf32>

    %C = linalg.matmul
           ins(%A_mark, %B_mark : tensor<4x4xf32>, tensor<4x4xf32>)
           outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>

    return %C : tensor<4x4xf32>
  }
}
