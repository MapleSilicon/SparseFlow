module {
  func.func @test_matmul(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8xf32>
    %result = linalg.matmul ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>) 
                outs(%cst : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %result : tensor<8x8xf32>
  }
}
