
---

## Static Sparsity Propagation Analysis (SPA) – v0.5

SparseFlow includes a **Sparsity Propagation Analysis (SPA)** pass that performs compile-time analysis of sparsity patterns:

**Features:**
- Reads N:M sparsity annotations from `linalg.matmul` (via `sparseflow-annotate-nm`)
- Converts N:M patterns into row-level sparsity masks (`sparseflow.spa_rowmask`)
- Propagates structural sparsity through arithmetic operations (add, mul, sub, div, max)
- Enables optimization opportunities based on proven-zero rows

**Quick demo:**
```bash
./scripts/run_spa_demo.sh
```

Or manually:
```bash
cd compiler/build
mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm),sparseflow-spa)' \
  ../tests/spa_nm_demo.mlir
```

**Example output:**
```mlir
linalg.matmul {sparseflow.m = 4, sparseflow.n = 2, 
               sparseflow.spa_rowmask = [true, true, false, false]}
arith.addf {sparseflow.spa_rowmask = [true, true, false, false]}
```

This shows that with 2:4 sparsity, rows 2 and 3 are provably zero and this information propagates through subsequent operations.

