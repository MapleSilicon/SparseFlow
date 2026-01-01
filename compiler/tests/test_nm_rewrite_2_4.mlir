module {
  func.func @test_2_4(%A: tensor<16x16xf32, {n = 2 : i32, m = 4 : i32}>, 
                      %B: tensor<16x16xf32>, 
                      %C: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<16x16xf32, {n = 2 : i32, m = 4 : i32}>, tensor<16x16xf32>)
           outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}
