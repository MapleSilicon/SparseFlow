#ifndef SPARSEFLOW_OPS_TD
#define SPARSEFLOW_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "SparseFlow/SparseFlowDialect.td"

def SparseFlow_SparseMatmulOp : Op<SparseFlow_Dialect, "sparse_matmul", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "SparseFlow masked matmul op";
  let description = [{
    Represents a sparse matmul using rowmask + colmask
    detected by SPA. This op will be lowered to a custom
    runtime call in the lowering pass.
  }];

  let arguments = (ins
    AnyShaped:$A,
    AnyShaped:$B,
    I32ArrayAttr:$rowmask,
    I32ArrayAttr:$colmask
  );

  let results = (outs AnyShaped:$C);

  let assemblyFormat = [{
    $A `,` $B attr-dict `:` type($A) `,` type($B) `->` type($C)
  }];
}

#endif // SPARSEFLOW_OPS_TD
