//===- SparseFlowDialect.td - SparseFlow dialect -----------*- tablegen -*-===//
//
// SparseFlow MLIR Dialect for sparse inference optimization
//
//===----------------------------------------------------------------------===//

#ifndef SPARSEFLOW_DIALECT
#define SPARSEFLOW_DIALECT

include "mlir/IR/OpBase.td"

def SparseFlow_Dialect : Dialect {
  let name = "sparseflow";
  let summary = "SparseFlow dialect for 2:4 sparse inference";
  let description = [{
    The SparseFlow dialect provides operations for expressing and optimizing
    2:4 structured sparse matrix operations on NVIDIA GPUs.
    
    Key features:
    - 2:4 sparse GEMM operations
    - Epilogue fusion (RELU, SiLU, GELU, etc.)
    - Automatic tile size optimization
    - Multi-operation fusion
  }];
  
  let cppNamespace = "::mlir::sparseflow";
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Base class for SparseFlow ops
class SparseFlow_Op<string mnemonic, list<Trait> traits = []> :
    Op<SparseFlow_Dialect, mnemonic, traits>;

#endif // SPARSEFLOW_DIALECT
