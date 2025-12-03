# SparseFlow SPA Progress — v0.5 (Current Status)

SparseFlow now includes a working **Static Sparsity Propagation Analysis (SPA)** system that reads N:M sparsity metadata, creates structural sparsity masks, and propagates them through compute graphs.

This document summarizes:

* ✔ What is **DONE**
* ❗ What is **IN PROGRESS**
* 🔥 What is **PLANNED / NOT DONE YET**
* 🎯 The **final goal** for SPA inside SparseFlow

---

# ✔ Completed (SPA v0.5 — Working Features)

## 1. Fully working MLIR pass plugin

* Compiles cleanly under MLIR 19
* Loaded via:
```bash
  mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so
```

## 2. N:M → Rowmask conversion (core feature)

`linalg.matmul` operations annotated by `sparseflow-annotate-nm` now produce structured masks.

Example:
```
sparseflow.n = 2
sparseflow.m = 4
```

Rowmask generated:
```
[true, true, false, false]
```

This is the **first real structured-sparsity semantic feature** in SparseFlow.

## 3. SPA propagation through arithmetic ops

SPA now propagates structural sparsity through:

* `arith.addf`
* `arith.mulf`
* `arith.subf`
* `arith.divf`
* `arith.maximumf`
* `linalg.matmul` (first-order only)

Propagation logic is stable, deterministic, and safe.

## 4. Support for user-defined sparsity

`sparseflow.mark` allows custom row masks:
```mlir
%s = "sparseflow.mark"(%A)
  { sparseflow.spa_rowmask = [true, false, false, true] }
```

And SPA correctly uses it downstream.

## 5. Comprehensive tests created

* `spa_v3_test.mlir`
* `spa_v4_test.mlir`
* `spa_v5_test.mlir`
* `spa_v6_matmul_2d.mlir`
* `spa_v4_comprehensive.mlir`
* `spa_nm_demo.mlir` (permanent test)
* `run_spa_demo.sh` (automation script)

These are now replicable for anyone.

## 6. Integration with AnnotateNmPass

Pipeline:
```bash
--pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm),sparseflow-spa)'
```

SPA reads the N:M annotations placed on matmuls.

---

# ❗ In Progress (v0.6 Targets)

These are partially implemented or close to done:

### 1. Column mask support (2D sparsity field)

Rowmask is complete.
Colmask exists as a placeholder but not fully implemented.

### 2. More stable matmul propagation

SPA currently uses:

* lhs rowmask
* but ignores rhs columnmask (because not implemented)

### 3. Better arithmetic semantics

`maximumf` propagation still needs separate handling for positive/negative dominance conditions.

### 4. Error handling + assertions

SPA should gracefully handle:

* unknown shapes
* mismatched ranks
* dynamic dimensions

This is partially implemented but not robust.

---

# 🔥 Not Done Yet (Major Future Features)

These are the "big ticket" items before SPA becomes production-level.

## 1. Full 2D Sparsity Domain

We need:
```
rowmask (size R)
colmask (size C)
```

SPA v0.5 only does rowmask.

## 2. Matmul reasoning using both masks

The correct propagation rule:
```
out[i, j] = sum_k lhs[i, k] * rhs[k, j]
```

Means:

* A row of output is zero if:

  * lhs row is zero OR
  * product over all columns cancels because of rhs sparsity

This is not implemented yet.

## 3. Tile-level masks (required for hardware runtime)

To support accelerators like:

* Ampere 2:4
* Hopper FP8 Sparse Tensor Cores
* Tenstorrent TT-Metal
* GroqChip

We need tile-level domain reasoning.

## 4. JSON metadata export for SPA

Export N:M + SPA masks:
```json
{
  "matmul0": {
    "rows": [1,1,0,0],
    "cols": [1,0,1,0],
    "nm": "2:4"
  }
}
```

Not implemented yet.

## 5. Integration with SparseFlow Runtime

Today the runtime ignores SPA metadata.
Final version will:

* skip zero rows
* skip zero blocks
* adjust FLOP counts
* generate structured-sparse kernels

---

# 🎯 Final Goal for SPA

Build a **full static sparsity compiler** that:

* Reads N:M from MLIR
* Learns effective sparsity patterns across ops
* Emits structured-sparse metadata and IR
* Enables real **2× hardware speedup**
* Avoids dynamic detection cost during runtime
* Matches or beats PyTorch/NVIDIA implementation quality

SPA is the brain of SparseFlow — the part that turns N:M into real acceleration.

---

# Summary Table

| Component                | Status         |
| ------------------------ | -------------- |
| N:M annotation pass      | **Done**       |
| SPA plugin               | **Done**       |
| Rowmask domain           | **Done**       |
| Rowmask propagation      | **Done**       |
| MarkOp passthrough       | **Done**       |
| Column mask domain       | ❗ Partial      |
| Matmul 2D propagation    | ❌ Not done     |
| Tile sparsity            | ❌ Not done     |
| JSON metadata export     | ❌ Not done     |
| Runtime integration      | ❌ Not done     |
| Full end-to-end pipeline | 🚧 In progress |

---

# Next Recommended Steps

1. **Commit this README** (delete the old placeholder text).
2. Add a `docs/` directory with examples + diagrams.
3. Start v0.6 with colmask + solver for matmul 2D propagation.
4. Plan how SPA connects into SparseFlow Runtime (v0.7).
5. Prepare a Twitter/X + LinkedIn post about SPA v0.5.

---

**Last Updated:** December 2024  
**Contributors:** Core SparseFlow Team  
**Status:** Active Development
