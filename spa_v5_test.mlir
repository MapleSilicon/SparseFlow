module {
  func.func @test(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {

    // Declare a sparse version of A by wrapping it in an op with the rowmask attribute
    %A_wrap = "sparseflow.mark"(%A)
      { sparseflow.spa_rowmask = [true, false, false, true] }
      : (tensor<4x4xf32>) -> tensor<4x4xf32>

    %zero = arith.constant dense<0.0> : tensor<4x4xf32>

    %add = arith.addf %A_wrap, %zero : tensor<4x4xf32>
    %mul = arith.mulf %add, %A_wrap : tensor<4x4xf32>
    %relu = arith.maximumf %mul, %zero : tensor<4x4xf32>
    %sub = arith.subf %relu, %zero : tensor<4x4xf32>
    %div = arith.divf %sub, %A_wrap : tensor<4x4xf32>

    return %div : tensor<4x4xf32>
  }
}
