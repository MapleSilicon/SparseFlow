module {
  func.func @main() -> tensor<32x32xf32> {
    %0 = arith.constant dense<1.0> : tensor<32x32xf32>
    %1 = arith.constant dense<2.0> : tensor<32x32xf32>
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = linalg.matmul {sparseflow.m = 4 : i32, sparseflow.n = 2 : i32}
          ins(%0, %1 : tensor<32x32xf32>, tensor<32x32xf32>)
          outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %3 : tensor<32x32xf32>
  }
}
