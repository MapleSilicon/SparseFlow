# SPA User Guide

## Installation

### Prerequisites

- LLVM/MLIR 19.x
- CMake 3.20+
- C++17 compiler

### Building SparseFlow with SPA
```bash
cd SparseFlow/compiler
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/lib/llvm-19 ..
make -j4
```

This produces `passes/SparseFlowPasses.so` with SPA included.

## Basic Usage

### Running SPA on MLIR
```bash
mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(sparseflow-spa)' \
  input.mlir
```

### With AnnotateNm Integration
```bash
mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm),sparseflow-spa)' \
  input.mlir
```

This runs:
1. AnnotateNm - adds N:M sparsity attributes
2. SPA - converts N:M to rowmasks and propagates

## Quick Start Examples

### Example 1: Simple Matmul

**Input (`simple.mlir`):**
```mlir
module {
  func.func @test(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %init = arith.constant dense<0.0> : tensor<4x4xf32>
    %C = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %C : tensor<4x4xf32>
  }
}
```

**Run:**
```bash
mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm),sparseflow-spa)' \
  simple.mlir
```

**Output:**
```mlir
%C = linalg.matmul {sparseflow.m = 4, sparseflow.n = 2,
                    sparseflow.spa_rowmask = [true, true, false, false]}
     ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
```

### Example 2: Propagation Through Arithmetic

**Input:**
```mlir
module {
  func.func @test(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %init = arith.constant dense<0.0> : tensor<4x4xf32>
    %C = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %D = arith.addf %C, %C : tensor<4x4xf32>
    %E = arith.mulf %D, %C : tensor<4x4xf32>
    return %E : tensor<4x4xf32>
  }
}
```

**Output shows propagation:**
```mlir
%C = linalg.matmul {sparseflow.spa_rowmask = [true, true, false, false]} ...
%D = arith.addf {sparseflow.spa_rowmask = [true, true, false, false]} ...
%E = arith.mulf {sparseflow.spa_rowmask = [true, true, false, false]} ...
```

### Example 3: Custom Sparsity Annotation

You can manually mark inputs as sparse:
```mlir
module {
  func.func @test(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // Mark A as having rows 0 and 3 non-zero, rows 1 and 2 zero
    %A_sparse = "sparseflow.mark"(%A)
      {sparseflow.spa_rowmask = [true, false, false, true]}
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
    
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    %B = arith.addf %A_sparse, %zero : tensor<4x4xf32>
    return %B : tensor<4x4xf32>
  }
}
```

**Output:**
```mlir
%A_sparse = "sparseflow.mark"(%A) 
  {sparseflow.spa_rowmask = [true, false, false, true]} ...
%B = arith.addf {sparseflow.spa_rowmask = [true, false, false, true]} ...
```

## Understanding Attributes

### `sparseflow.n` and `sparseflow.m`

Added by `AnnotateNm` pass:
- `n` = number of non-zero values
- `m` = total values in pattern
- Example: `n=2, m=4` means 2:4 sparsity (50% sparse)

### `sparseflow.spa_rowmask`

Added by SPA pass:
- Array of booleans, one per row
- `true` = row may be non-zero
- `false` = row is provably zero

**Example:**
```mlir
{sparseflow.spa_rowmask = [true, true, false, false]}
```
Rows 0 and 1 may have data, rows 2 and 3 are zero.

## Supported Operations

| Operation | Propagation Rule | Example |
|-----------|------------------|---------|
| `linalg.matmul` | LHS row → output row | If A[i] is zero, C[i] is zero |
| `arith.addf` | Union (OR) | Zero if both inputs zero |
| `arith.subf` | Union (OR) | Zero if both inputs zero |
| `arith.mulf` | Intersection (AND) | Zero if either input zero |
| `arith.divf` | Union (OR) | Conservative |
| `arith.maximumf` | Pass-through | Zero rows stay zero |
| `linalg.transpose` | Swap rows/cols | (v0.6+) |
| `linalg.reduce` | Preserve pattern | Row reduction |

## Tips and Best Practices

### 1. Run AnnotateNm First

Always run AnnotateNm before SPA to get N:M information:
```bash
--pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm),sparseflow-spa)'
```

### 2. Check Intermediate Results

Use `mlir-opt` to inspect IR at each stage:
```bash
# After AnnotateNm only
mlir-opt-19 ... --pass-pipeline='...(sparseflow-annotate-nm)' input.mlir

# After both
mlir-opt-19 ... --pass-pipeline='...(sparseflow-annotate-nm),sparseflow-spa)' input.mlir
```

### 3. Verify Propagation

Look for `spa_rowmask` appearing on downstream operations. If it doesn't propagate, check:
- Are operations supported?
- Are tensor ranks correct (must be 2D)?
- Are dimensions known (not dynamic)?

### 4. Use the Demo Script

Quick validation:
```bash
./scripts/run_spa_demo.sh
```

## Troubleshooting

### Problem: No rowmask attributes appear

**Solution:** Check that:
1. Pass plugin loads correctly
2. Operations are 2D tensors (4D/5D not yet supported)
3. AnnotateNm runs first (for N:M integration)

### Problem: Rowmask is all `true`

**Possible causes:**
- Inputs have no sparsity info (expected behavior)
- N:M attributes missing
- Operation not reaching matmul

**Solution:** Add explicit `sparseflow.mark` to test, or verify AnnotateNm ran.

### Problem: Propagation stops at certain ops

**Cause:** Unsupported operation type

**Solution:** Check [Supported Operations](#supported-operations) table, or file an issue for new op support.

## Advanced Usage

### Programmatic Access

In your own MLIR pass, read SPA results:
```cpp
if (auto attr = op->getAttrOfType<ArrayAttr>("sparseflow.spa_rowmask")) {
  for (auto elem : attr) {
    bool isNonZero = elem.cast<BoolAttr>().getValue();
    // Use sparsity info for optimization
  }
}
```

### Custom Propagation Rules

See [Architecture](architecture.md#extension-points) for adding new operations.

## Next Steps

- See [Examples](examples/) for more complex scenarios
- Read [Architecture](architecture.md) to understand internals
- Check [SPA_STATUS.md](../../SPA_STATUS.md) for roadmap

