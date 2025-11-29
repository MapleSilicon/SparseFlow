# SparseFlow Quick Demo

## What is SparseFlow?

SparseFlow is an MLIR-based compiler pipeline for N:M sparsity.

- Annotates `linalg.matmul` operations with N:M sparsity patterns
- Exports structured JSON metadata for downstream tools
- Analyzes dense vs sparse FLOPs (e.g., 2:4 sparsity)

Given an MLIR module, SparseFlow tells you:
- which matmuls use N:M sparsity
- their input/output tensor shapes
- FLOP reduction achieved from sparsity

## Prerequisites

- LLVM/MLIR 19 with `mlir-opt-19` in `PATH`
- CMake and Ninja
- Repo checked out at `~/src/SparseFlow`

## Build & Demo

From the repo root:

```bash
cd ~/src/SparseFlow
./scripts/run_sparseflow_demo.sh
