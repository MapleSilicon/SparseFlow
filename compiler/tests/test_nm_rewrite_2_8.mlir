module {
  func.func @test_2_8(%A: tensor<32x32xf32, {n = 2 : i32, m = 8 : i32}>, 
                      %B: tensor<32x32xf32>, 
                      %C: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<32x32xf32, {n = 2 : i32, m = 8 : i32}>, tensor<32x32xf32>)
           outs(%C : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
