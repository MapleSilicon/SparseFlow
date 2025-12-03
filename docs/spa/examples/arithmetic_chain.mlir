// Example 2: Sparsity propagation through arithmetic operations
// Shows how rowmasks flow through add, mul, and max operations

module {
  func.func @arithmetic_chain(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %init = arith.constant dense<0.0> : tensor<8x8xf32>
    
    // Matmul creates sparsity pattern
    %C = linalg.matmul ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>)
                       outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    
    // Add preserves zero rows (union)
    %D = arith.addf %C, %C : tensor<8x8xf32>
    
    // Multiply intersects patterns (AND)
    %E = arith.mulf %D, %C : tensor<8x8xf32>
    
    // ReLU (max with zero) preserves zeros
    %F = arith.maximumf %E, %init : tensor<8x8xf32>
    
    return %F : tensor<8x8xf32>
  }
}
