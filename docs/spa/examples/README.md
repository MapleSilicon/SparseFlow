# SPA Examples

This directory contains example MLIR files demonstrating SPA features.

## Running Examples

All examples can be run with:
```bash
cd SparseFlow/compiler/build
mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm),sparseflow-spa)' \
  ../../docs/spa/examples/<example>.mlir
```

## Example Files

- **`basic_matmul.mlir`** - Simple matmul with N:M annotation
- **`arithmetic_chain.mlir`** - Propagation through multiple operations
- **`custom_annotation.mlir`** - Manual sparsity marking with `sparseflow.mark`

## Expected Outputs

### basic_matmul.mlir

With 2:4 sparsity (default), you should see:
```mlir
sparseflow.spa_rowmask = [true, true, false, false, true, true, false, false]
```

### arithmetic_chain.mlir

All operations should carry the same rowmask from the matmul:
```mlir
linalg.matmul {sparseflow.spa_rowmask = [...]}
arith.addf {sparseflow.spa_rowmask = [...]}
arith.mulf {sparseflow.spa_rowmask = [...]}
arith.maximumf {sparseflow.spa_rowmask = [...]}
```

### custom_annotation.mlir

Custom pattern `[true, false, false, true]` should propagate:
```mlir
"sparseflow.mark" {sparseflow.spa_rowmask = [true, false, false, true]}
arith.addf {sparseflow.spa_rowmask = [true, false, false, true]}
arith.mulf {sparseflow.spa_rowmask = [true, false, false, true]}
```

