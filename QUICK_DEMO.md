
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


## SPA v0.6 – 2D Sparsity (Rows + Cols) Demo

SparseFlow includes an experimental SPA (Sparsity Propagation Analysis) pass that tracks **row** and **column** sparsity through operations and exports the result as JSON.

### What it does

For a simple 4x4 `linalg.matmul` with the following compile-time masks:

- Input A: `rowmask = [true, false, true, false]` → 50% of rows are dead
- Input B: `colmask = [true, true, false, false]` → 50% of columns are dead

SPA computes:

- Output `linalg.matmul` also has:
  - `rowmask = [true, false, true, false]`
  - `colmask = [true, true, false, false]`
- Effective density ≈ **25%** → ≈ **75% sparsity**
- Ideal speedup if hardware/runtime fully exploits the masks: ≈ **4×** fewer MACs.

### How to run the demo

From the repo root:

```bash
./run_spa_v06_demo.sh


---

## SPA v0.6 – 2D Sparsity (Rows + Cols) Demo

SparseFlow includes **SPA (Sparsity Propagation Analysis)** that tracks **row** and **column** sparsity through operations and exports the result as JSON.

### What it does

For a simple 4×4 `linalg.matmul` with compile-time masks:

- **Input A:** `rowmask = [true, false, true, false]` → 50% of rows are dead
- **Input B:** `colmask = [true, true, false, false]` → 50% of columns are dead

SPA computes:

- **Output matmul** has:
  - `rowmask = [true, false, true, false]` (inherited from A)
  - `colmask = [true, true, false, false]` (inherited from B)
- **Effective density:** 25% (only 4 out of 16 elements needed)
- **Effective sparsity:** 75%
- **Ideal speedup:** **~4× fewer MACs**

### How to run the demo

From the repo root:
```bash
./run_spa_v06_demo.sh
```

This will:
1. Rebuild `SparseFlowPasses.so`
2. Run `sparseflow-spa` + `sparseflow-spa-export` on `test_spa_v6_full_2d.mlir`
3. Print the resulting `spa_sparsity.json`

**Example JSON output:**
```json
{
  "operations": [
    {
      "id": 2,
      "name": "linalg.matmul",
      "total_rows": 4,
      "total_cols": 4,
      "zero_rows": 2,
      "zero_cols": 2,
      "row_sparsity_pct": 50,
      "col_sparsity_pct": 50,
      "rowmask": [true, false, true, false],
      "colmask": [true, true, false, false]
    }
  ]
}
```

### Analyze the results

To see sparsity analysis and speedup estimates:
```bash
./analyze_spa_json.py
./estimate_spa_speedup.py
```

**Output:**
```
Effective density : 25.0%
Effective sparsity: 75.0%
Ideal relative speedup: ~4.00x
```

### What this means

**Bottom line:** SparseFlow's SPA pass proves that for structured sparsity patterns, we can statically eliminate **~75% of computation** in matmul operations. The exported masks can drive hardware/runtime optimizations for **up to 4× speedup**.

**Current status:** Static analysis complete ✅ | JSON export working ✅ | Runtime integration: TODO

