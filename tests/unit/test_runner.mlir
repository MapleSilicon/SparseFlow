module {
  func.func private @sparse_matmul_2_4(tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  
  func.func @test() {
    // Create test matrices
    %A = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0],
                                [13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf32>
    
    %B = arith.constant dense<[[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xf32>
    
    // Call sparse matmul
    %result = func.call @sparse_matmul_2_4(%A, %B) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    
    return
  }
}
