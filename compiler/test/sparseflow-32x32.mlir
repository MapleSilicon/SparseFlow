module {
  func.func @large_matmul() -> tensor<32x32xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<32x32xf32>
    %0 = tensor.empty() : tensor<32x32xf32>
    %1 = linalg.matmul
         {sparseflow.m = 4 : i32, sparseflow.n = 2 : i32}
         ins(%cst, %cst_0 : tensor<32x32xf32>, tensor<32x32xf32>)
         outs(%0 : tensor<32x32xf32>)
         -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
