module {
  func.func @test_4_16(%A: tensor<64x64xf32> {n = 4 : i32, m = 16 : i32},
                      %B: tensor<64x64xf32>,
                      %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>)
           outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
