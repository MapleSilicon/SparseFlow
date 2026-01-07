module {
  func.func private @sparse_matmul_2_4(tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  
  func.func @test() {
    %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
    
    %result = func.call @sparse_matmul_2_4(%A, %B) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    
    return
  }
}
