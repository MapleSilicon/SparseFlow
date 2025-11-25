module {
  func.func @matmul_test(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>) -> tensor<128x64xf32> {
    %result = linalg.matmul ins(%A, %B: tensor<128x256xf32>, tensor<256x64xf32>)
                          -> tensor<128x64xf32>
    return %result : tensor<128x64xf32>
  }

  func.func @another_matmul(%A: tensor<64x128xf32>, %B: tensor<128x32xf32>) -> tensor<64x32xf32> {
    %result = linalg.matmul ins(%A, %B: tensor<64x128xf32>, tensor<128x32xf32>)
                          -> tensor<64x32xf32>
    return %result : tensor<64x32xf32>
  }
}
