//===- Ops.td - SparseFlow dialect operations ---------------*- tablegen -*-===//
//
// SparseFlow custom operations for sparse computation
//
//===----------------------------------------------------------------------===//

#ifndef SPARSEFLOW_OPS
#define SPARSEFLOW_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SparseFlow_Dialect : Dialect {
  let name = "sparseflow";
  let cppNamespace = "::sparseflow";
  let summary = "SparseFlow dialect for sparse computation";
  let description = [{
    Custom operations for SparseFlow sparse computation pipeline.
  }];
}

def SparseFlow_SparseMatmulOp : Op<SparseFlow_Dialect, "sparse_matmul",
    [Pure]> {

  let summary = "SparseFlow masked matrix multiplication";

  let description = [{
    Represents a matrix multiplication that uses rowmask and colmask
    produced by the SPA (Sparsity Propagation Analysis) pass.
    
    This operation is lowered to a call to the SparseFlow runtime
    which implements an optimized sparse kernel.
    
    Example:
```mlir
    %C = sparseflow.sparse_matmul %A, %B 
      rowmask = [1, 0, 1, 0]
      colmask = [1, 1, 0, 0]
      : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
```
  }];

  let arguments = (ins
    AnyRankedTensor:$A,
    AnyRankedTensor:$B,
    ArrayAttr:$rowmask,
    ArrayAttr:$colmask
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $A `,` $B attr-dict `:` type($A) `,` type($B) `->` type($result)
  }];
}

#endif // SPARSEFLOW_OPS
