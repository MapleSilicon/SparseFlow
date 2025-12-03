// Example 3: Manual sparsity annotation
// Use sparseflow.mark to explicitly set rowmasks

module {
  func.func @custom_sparse(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // Manually mark A with custom sparsity pattern
    // Rows 0 and 3 are non-zero, rows 1 and 2 are zero
    %A_sparse = "sparseflow.mark"(%A)
      {sparseflow.spa_rowmask = [true, false, false, true]}
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    
    // Operations preserve the pattern
    %B = arith.addf %A_sparse, %zero : tensor<4x4xf32>
    %C = arith.mulf %B, %A_sparse : tensor<4x4xf32>
    
    return %C : tensor<4x4xf32>
  }
}
