# SparseFlow

SparseFlow is an experimental MLIR-based compiler pipeline for N:M structured sparsity.

## Components

- `compiler/`
  - MLIR pass plugin (`SparseFlowPasses.so`)
  - Passes:
    - `sparseflow-annotate-nm`
    - `sparseflow-verify-pattern`
    - `sparseflow-generate-mask`
    - `sparseflow-simple-gpu-lowering`
    - `sparseflow-export-metadata`
  - Tests in `compiler/test/`

- `sim/`
  - Simple SPMM / kernel simulation in C++.

- `benchmarks/`
  - Scripts to estimate speedups from exported JSON metadata.

## Quick build

```bash
cd compiler
mkdir -p build
cd build

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir

ninja SparseFlowPasses

