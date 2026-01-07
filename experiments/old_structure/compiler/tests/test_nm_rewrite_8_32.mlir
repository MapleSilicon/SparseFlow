module {
  func.func @test_8_32(%A: tensor<64x64xf32, {n = 8 : i32, m = 32 : i32}>, 
                       %B: tensor<64x64xf32>, 
                       %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<64x64xf32, {n = 8 : i32, m = 32 : i32}>, tensor<64x64xf32>)
           outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
