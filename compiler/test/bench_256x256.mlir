module {
  func.func @matmul_256(%arg0: tensor<256x256xf32>, 
                            %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<256x256xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
    %result = linalg.matmul ins(%arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32>)
                           outs(%filled : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %result : tensor<256x256xf32>
  }
}
