module {
  func.func @test(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %B = arith.addf %A, %A : tensor<4x4xf32>
    return %B : tensor<4x4xf32>
  }
}
