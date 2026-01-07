module {
  func.func @matmul_512(%arg0: tensor<512x512xf32>, 
                            %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<512x512xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
    %result = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>)
                           outs(%filled : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %result : tensor<512x512xf32>
  }
}
