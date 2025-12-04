#ifndef SPARSEFLOW_DIALECT_TD
#define SPARSEFLOW_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def SparseFlow_Dialect : Dialect {
  let name = "sparseflow";
  let cppNamespace = "sparseflow";

  let summary = "SparseFlow sparsity-aware operations";
  let description = [{
    This dialect defines SparseFlow operations for static sparsity
    propagation, masked matmuls, and related transformations.
  }];

  let dependentDialects = [];
}

#endif // SPARSEFLOW_DIALECT_TD
