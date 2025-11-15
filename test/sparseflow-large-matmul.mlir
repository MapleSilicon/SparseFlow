// RUN: mlir-opt-19 -load-pass-plugin %llvmshlibdir/libSparseFlowPasses.so \
// RUN:   -pass-pipeline='builtin.module(func.func(
// RUN:     sparseflow-annotate-nm{n=2 m=4},
// RUN:     sparseflow-verify-pattern,
// RUN:     sparseflow-generate-mask,
// RUN:     sparseflow-sparse-compute
// RUN:   ))' %s | FileCheck %s

module {
  func.func @large_matmul() -> tensor<64x64xf32> {
    %lhs = arith.constant dense<1.0> : tensor<64x64xf32>
    %rhs = arith.constant dense<2.0> : tensor<64x64xf32>
    %init = tensor.empty() : tensor<64x64xf32>
    %result = linalg.matmul ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                          outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }
}

// CHECK-LABEL: func.func @large_matmul
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.remsi
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK: tensor.insert
// CHECK-NOT: linalg.matmul
