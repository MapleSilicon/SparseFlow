# SparseFlow v0.2.0 – Generalized N:M Sparse Runtime + Rewrite Pass

## Overview

SparseFlow v0.2.0 is the first fully validated release that combines:

- A **generalized N:M sparse runtime** (1:4, 2:4, 2:8, 4:16, 8:32)
- A **compiler rewrite pass** that emits calls to the correct sparse kernels
- End-to-end validation with MLIR tests, runtime tests, and benchmarks

This is the first version that is realistically **benchmarked, reproducible, and investor-ready**.

---

## Key Features

### 1. Generalized N:M Sparse Runtime

- Supports 5 structured N:M patterns:
  - `1:4`, `2:4`, `2:8`, `4:16`, `8:32`
- Each implemented as its own kernel:
  - `sparse_matmul_1_4`
  - `sparse_matmul_2_4`
  - `sparse_matmul_2_8`
  - `sparse_matmul_4_16`
  - `sparse_matmul_8_32`
- Optimized for CPU with block-contiguous storage and branch-free inner loops

### 2. Compiler Integration (MLIR)

- SPA (Sparsity Propagation Analysis) tracks 2D sparsity (rows + cols)
- N:M metadata is propagated and exported
- `SparseMatmulRewritePass`:
  - Reads N:M encoding from tensor types
  - Generates dynamic kernel calls: `func.call @sparse_matmul_N_M(...)`
- 5 MLIR tests for all patterns validate correct code generation

### 3. Validation & Tooling

- `validate_everything.sh` - Comprehensive system validation
- `run_nm_rewrite_tests.sh` - MLIR rewrite pass verification
- Benchmark tools for performance measurement

---

## Benchmark Summary (CPU)

**Honest Performance Claims:**
- **9-20× validated speedup** across N:M patterns on CPU
- Tested on matrix sizes: 256×256 through 2048×2048
- 2:4 pattern (50% density): ~9-12× speedup
- Lower density patterns (1:4, 2:8, 4:16, 8:32): ~12-20× speedup

These are **real wall-clock measurements**, not theoretical FLOP reductions.

See [`BENCHMARK_RESULTS_v0.2.md`](./BENCHMARK_RESULTS_v0.2.md) for full details.

---

## How to Build and Run v0.2.0

### 1. Build Compiler
```bash
cd compiler
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/lib/llvm-19 ..
make -j8
```

### 2. Build Runtime
```bash
cd ../../runtime
mkdir -p build && cd build
cmake ..
make -j8
```

### 3. Run Validation
```bash
cd ~/SparseFlow/SparseFlow
./validate_everything.sh
```

Expected: **18 passed, 0 failed**

### 4. Run Benchmarks
```bash
cd runtime/build
./benchmark_nm_runtime
```

---

## What's New in v0.2.0

- ✅ Generalized N:M runtime (5 patterns)
- ✅ Dynamic kernel selection in compiler pass
- ✅ Comprehensive MLIR integration tests
- ✅ Validated benchmarks (9-20× speedup)
- ✅ Professional documentation
- ✅ CI/CD pipeline

---

## Roadmap

* **v0.3 (Q1 2026)** – GPU acceleration (CUDA / Tensor Cores)
* **v0.4 (Q2 2026)** – PyTorch integration (`torch.compile`)
* **v0.5 (Q3 2026)** – Production deployment

---

## Technical Details

**Architecture:**
- MLIR-based compiler with custom passes
- C++ runtime with template-based kernel instantiation
- Inline dictionary encoding for tensor metadata

**Compatibility:**
- MLIR/LLVM 19
- Ubuntu 24.04
- CPU-optimized (GPU coming in v0.3)

**Repository:** https://github.com/MapleSilicon/SparseFlow

