module {
  func.func @test(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    
    // Basic ops
    %add = arith.addf %A, %zero : tensor<4x4xf32>
    %mul = arith.mulf %add, %A : tensor<4x4xf32>
    
    // ReLU approximation (max with zero)
    %relu = arith.maximumf %mul, %zero : tensor<4x4xf32>
    
    // More arithmetic
    %sub = arith.subf %relu, %zero : tensor<4x4xf32>
    %div = arith.divf %sub, %A : tensor<4x4xf32>
    
    return %div : tensor<4x4xf32>
  }
}
