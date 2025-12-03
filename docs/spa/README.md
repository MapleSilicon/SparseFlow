# SparseFlow SPA (Sparsity Propagation Analysis)

## Overview

SPA is a static analysis pass that tracks and propagates sparsity information through MLIR operations. It enables compile-time optimization by identifying which rows/columns of tensors are provably zero.

## Key Features

- **N:M Sparsity Integration**: Reads structured sparsity from pruning passes
- **Row-level Analysis**: Tracks which matrix rows are zero
- **Propagation**: Flows sparsity through arithmetic and linear algebra operations
- **Optimization Ready**: Metadata enables kernel selection and computation skipping

## Quick Links

- [Architecture Overview](architecture.md)
- [User Guide](user_guide.md)
- [Examples](examples/)
- [API Reference](api_reference.md)

## How It Works
```
Input IR (with N:M annotations)
           ↓
    SPA Analysis Pass
           ↓
  Annotated IR (with rowmasks)
           ↓
   Optimization Passes
           ↓
    Optimized Code
```

## Example

**Before SPA:**
```mlir
%C = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
%D = arith.addf %C, %C : tensor<4x4xf32>
```

**After SPA (with 2:4 sparsity):**
```mlir
%C = linalg.matmul {sparseflow.spa_rowmask = [true, true, false, false]}
     ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
%D = arith.addf {sparseflow.spa_rowmask = [true, true, false, false]}
     %C, %C : tensor<4x4xf32>
```

Now the compiler knows rows 2 and 3 are zero!

## Getting Started

See [User Guide](user_guide.md) for detailed instructions.

Quick start:
```bash
./scripts/run_spa_demo.sh
```
