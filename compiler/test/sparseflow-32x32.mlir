module {
  func.func @matmul_32(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x32xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    %result = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
                           outs(%filled : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}
