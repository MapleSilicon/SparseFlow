# SparseFlow v0.1 — 2:4 Structured Sparsity Compiler Prototype

Tag: \`v0.1.0\`  
Date: December 2025

This is the first **end-to-end** SparseFlow release that connects:

- MLIR frontend
- SPA (Sparsity Propagation Analysis)
- Rewrite pass
- Lowering to LLVM
- JIT execution via \`ExecutionEngine\`
- Custom C++ / OpenMP sparse runtime

for the **2:4 structured sparsity pattern on CPU**.

---

## 🚀 Summary

SparseFlow v0.1.0 is a working prototype of a sparse MLIR-based compiler that:

- Detects sparse matmuls with SPA
- Rewrites them to calls into a custom sparse runtime kernel (\`sparse_matmul_2_4\`)
- Lowers the transformed IR to LLVM dialect
- JIT-compiles and executes via \`sparseflow-runner\`
- Achieves **~4× speedups** over a dense baseline at 50% 2:4 sparsity on medium–large matmuls

This is a **CPU / OpenMP** prototype, focused on getting a correct and measurable 2:4 pipeline running end-to-end.

---

## ✅ Key Features in v0.1.0

### 1. SPA (Sparsity Propagation Analysis)

- Analyzes tensor-level MLIR to find sparse opportunities.
- Emits \`sparseflow.mark\` metadata with:
  - Row-wise sparsity masks
  - Column-wise sparsity masks
- Drives the rewrite pass to target only the sparse matmuls.

### 2. Sparse Matmul Rewrite Pass

- Pass: \`sparseflow-rewrite-matmul\`
- Replaces SPA-marked \`linalg.matmul\` with calls to:
  - \`sparse_matmul_2_4\` (runtime kernel)
- Includes structural checks (shapes, types) before rewriting.
- Leaves unsupported matmuls untouched (safe fallback).

### 3. C++ / OpenMP Runtime Kernel

- Kernel: \`sparse_matmul_2_4\`
- Assumes 2:4 structured sparsity in the K dimension.
- Skips 75% of the FLOPs for 50% 2:4 sparsity.
- Uses OpenMP for parallelism.
- Validated against a dense reference implementation.

### 4. MLIR → LLVM → JIT Integration

- Lowering pipeline from tensor/memref IR to LLVM dialect.
- \`sparseflow-runner\`:
  - Loads MLIR
  - Registers Builtin + LLVM dialect translations
  - Loads \`libsparseflow_runtime.so\`
  - Registers \`sparse_matmul_2_4\` symbol
  - JITs and executes the entry function

### 5. Demo + Test Harness

- \`run_sparseflow_demo.sh\`:
  - JIT correctness tests
  - Performance benchmark suite
  - Full compiler pipeline demo (SPA → rewrite → lowering → JIT)

---

## 🧪 Correctness: JIT vs Dense Reference

To validate correctness of the sparse pipeline, we compare:

- JIT-executed sparse matmul (via SparseFlow)
- Dense reference implementation

Run:

```bash
cd ~/src/SparseFlow/compiler/build
./test_jit_correctness

