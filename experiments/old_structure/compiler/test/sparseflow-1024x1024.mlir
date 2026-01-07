module {
  func.func @matmul_1024(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1024x1024xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %result = linalg.matmul ins(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                           outs(%filled : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %result : tensor<1024x1024xf32>
  }
}
