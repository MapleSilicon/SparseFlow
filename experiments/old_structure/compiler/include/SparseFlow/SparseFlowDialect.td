// SparseFlow Dialect
// Defines operations for sparse tensor processing

#ifndef SPARSEFLOW_DIALECT
#define SPARSEFLOW_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SparseFlow_Dialect : Dialect {
  let name = "sparseflow";
  let cppNamespace = "::mlir::sparseflow";
  let description = [{
    The SparseFlow dialect provides operations and types for sparse tensor
    processing and sparsity-aware optimizations.
  }];
  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

// Base class for SparseFlow operations
class SparseFlow_Op<string mnemonic, list<Trait> traits = []> :
    Op<SparseFlow_Dialect, mnemonic, traits>;

// Mark operation - marks a tensor with sparsity information
def MarkOp : SparseFlow_Op<"mark"> {
  let summary = "Mark a tensor with sparsity metadata";
  let description = [{
    The `mark` operation annotates a tensor with sparsity information
    such as row/column masks for structured sparsity.
  }];

  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<I1ArrayAttr>:$rowmask,
    OptionalAttr<I1ArrayAttr>:$colmask
  );

  let results = (outs
    AnyTensor:$output
  );

  let assemblyFormat = [{
    $input (`rowmask` `=` $rowmask^)? (`colmask` `=` $colmask^)?
    `:` type($input) `->` type($output)
  }];
}

// Sparse matmul operation
def SparseMatmulOp : SparseFlow_Op<"sparse_matmul"> {
  let summary = "Sparse matrix multiplication";
  let description = [{
    Performs matrix multiplication with sparsity-aware computation.
    Can skip zero rows/columns based on masks.
  }];

  let arguments = (ins
    AnyTensor:$A,
    AnyTensor:$B,
    OptionalAttr<I1ArrayAttr>:$rowmask,
    OptionalAttr<I1ArrayAttr>:$colmask
  );

  let results = (outs
    AnyTensor:$C
  );

  let assemblyFormat = [{
    $A `,` $B (`rowmask` `=` $rowmask^)? (`colmask` `=` $colmask^)?
    `:` type($A) `,` type($B) `->` type($C)
  }];
}

#endif // SPARSEFLOW_DIALECT
