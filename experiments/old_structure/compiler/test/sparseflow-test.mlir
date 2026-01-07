// RUN: mlir-opt-19 -load-pass-plugin %llvmshlibdir/SparseFlowPasses%shlibext \
// RUN:   -pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm{n=2 m=4},sparseflow-export-metadata))' \
// RUN:   %s 2>&1 | FileCheck %s

module {
  func.func @matmul_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK: SPARSEFLOW_METADATA:
// CHECK: "op_type": "linalg.matmul"
// CHECK: "nm_pattern": "2:4"
