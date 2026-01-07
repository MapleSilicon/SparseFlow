module {
  func.func @main() -> tensor<64x64xf32> {
    %0 = arith.constant dense<1.0> : tensor<64x64xf32>
    %1 = arith.constant dense<2.0> : tensor<64x64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = linalg.matmul {sparseflow.m = 4 : i32, sparseflow.n = 2 : i32}
          ins(%0, %1 : tensor<64x64xf32>, tensor<64x64xf32>)
          outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }
}
