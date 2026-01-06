//===- SparseFlowOps.td - SparseFlow operations ------------*- tablegen -*-===//

#ifndef SPARSEFLOW_OPS
#define SPARSEFLOW_OPS

include "SparseFlowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Sparse GEMM Operation
//===----------------------------------------------------------------------===//

def SparseFlow_SparseGemmOp : SparseFlow_Op<"sparse_gemm",
    [Pure, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "2:4 sparse matrix multiplication";
  let description = [{
    Performs sparse matrix multiplication C = A @ B where:
    - A is M×K dense matrix (FP16)
    - B is K×N sparse matrix in 2:4 format (compressed)
    - C is M×N dense output (FP16)
    
    Example:
```mlir
    %result = sparseflow.sparse_gemm %A, %Bc {
      tile_m = 128 : i32,
      tile_n = 128 : i32,
      tile_k = 64 : i32
    } : (tensor<4096x4096xf16>, tensor<4096x4096xf16, sparse<2:4>>) 
        -> tensor<4096x4096xf16>
```
  }];
  
  let arguments = (ins
    AnyTensor:$A,
    AnyTensor:$Bc,
    OptionalAttr<I32Attr>:$tile_m,
    OptionalAttr<I32Attr>:$tile_n,
    OptionalAttr<I32Attr>:$tile_k
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = [{
    $A `,` $Bc attr-dict `:` 
    `(` type($A) `,` type($Bc) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Fused Sparse GEMM with Epilogue
//===----------------------------------------------------------------------===//

def SparseFlow_FusedSparseGemmOp : SparseFlow_Op<"fused_sparse_gemm",
    [Pure, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Sparse GEMM with fused epilogue operation";
  let description = [{
    Performs sparse GEMM with epilogue fusion in a single kernel.
    
    Supported epilogues:
    - "none": No epilogue
    - "relu": max(0, x)
    - "silu": x * sigmoid(x)
    - "gelu": GELU activation
    - "bias": x + bias
    - "bias_relu": max(0, x + bias)
    
    Example:
```mlir
    %result = sparseflow.fused_sparse_gemm %A, %Bc {
      epilogue = "silu",
      tile_m = 128 : i32,
      tile_n = 128 : i32,
      tile_k = 64 : i32
    } : (tensor<4096x4096xf16>, tensor<4096x4096xf16, sparse<2:4>>) 
        -> tensor<4096x4096xf16>
```
  }];
  
  let arguments = (ins
    AnyTensor:$A,
    AnyTensor:$Bc,
    OptionalAttr<AnyTensor>:$bias,
    StrAttr:$epilogue,
    OptionalAttr<I32Attr>:$tile_m,
    OptionalAttr<I32Attr>:$tile_n,
    OptionalAttr<I32Attr>:$tile_k
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = [{
    $A `,` $Bc attr-dict `:` 
    `(` type($A) `,` type($Bc) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Compressed Tensor Type
//===----------------------------------------------------------------------===//

def SparseFlow_CompressOp : SparseFlow_Op<"compress",
    [Pure]> {
  let summary = "Compress dense tensor to 2:4 sparse format";
  let description = [{
    Compresses a dense tensor that follows 2:4 sparsity pattern
    into the compressed representation used by sparse Tensor Cores.
    
    Example:
```mlir
    %compressed = sparseflow.compress %dense : 
      tensor<4096x4096xf16> -> tensor<4096x4096xf16, sparse<2:4>>
```
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

#endif // SPARSEFLOW_OPS
