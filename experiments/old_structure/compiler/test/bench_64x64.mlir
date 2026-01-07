module {
  func.func @matmul_64(%arg0: tensor<64x64xf32>, 
                            %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x64xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %result = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
                           outs(%filled : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }
}
