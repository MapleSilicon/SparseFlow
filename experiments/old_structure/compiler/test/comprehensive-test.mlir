// RUN: mlir-opt-19 -load-pass-plugin %llvmshlibdir/SparseFlowPasses%shlibext \
// RUN:   -pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm{n=2 m=4},sparseflow-verify-pattern,sparseflow-export-metadata))' \
// RUN:   %s 2>&1 | FileCheck %s

module {
  func.func @matmul_4x4(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK: remark: Annotated with sparseflow.nm = "2:4"
// CHECK: remark: validated 2:4 sparsity pattern
// CHECK: SPARSEFLOW_METADATA: {"metadata":{"total_ops":1,"version":"1.0"},"operations":[{"location":{[[FILE_LINE_COL:.*]]},"nm_pattern":"2:4","op_type":"linalg.matmul","operands":[{"role":"input","type":"tensor<4x4xf32>"},{"role":"input","type":"tensor<4x4xf32>"},{"role":"output","type":"tensor<4x4xf32>"}]}]}
