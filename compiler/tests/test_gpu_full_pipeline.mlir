module {
  func.func @test_gpu(%A: tensor<16x16xf32, {n = 2 : i32, m = 4 : i32}>, 
                      %B: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %c0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %0 = linalg.matmul
        ins(%A, %B : tensor<16x16xf32, {n = 2 : i32, m = 4 : i32}>, tensor<16x16xf32>)
        outs(%c0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}
