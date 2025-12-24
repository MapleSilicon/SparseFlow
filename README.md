# SparseFlow

SparseFlow is an MLIR-based compiler framework for **preserving, validating, and executing structured sparsity (N:M)** across the machine learning compilation pipeline.

SparseFlow is developed and maintained by **Maple Silicon Inc.**  
🌐 https://maplesilicon.co

---

## Overview

Modern AI models increasingly exhibit **structured sparsity** (e.g. 2:4), but most software stacks still lower and execute these models as dense workloads. This leads to:

- Wasted memory bandwidth
- Higher power consumption
- Underutilized silicon
- Fragile, backend-specific optimizations

SparseFlow addresses this problem at the **compiler level**, treating sparsity as a **first-class, verifiable property** rather than an afterthought.

---

## Design Philosophy

SparseFlow is built around three core principles:

### 1. Compiler-Level Sparsity Contracts
Sparsity is explicitly represented and validated during compilation, rather than inferred implicitly by backend kernels.

### 2. Correctness Before Performance
SparseFlow prioritizes **end-to-end correctness and verifiability**.  
Performance tuning is a later phase, not the starting point.

### 3. Safe Fallbacks
When sparsity constraints are not met, execution safely falls back to dense paths — avoiding silent correctness failures.

---

## What SparseFlow Does

SparseFlow provides a compiler pipeline discipline for structured sparsity, including:

- Explicit representation of structured sparsity patterns (N:M, e.g. 2:4)
- MLIR-based intermediate representations
- Verified lowering and transformation passes
- CPU and GPU execution paths
- End-to-end sparsity metadata propagation and validation

Rather than relying on opaque, backend-specific kernel behavior, SparseFlow ensures sparsity is either **preserved explicitly** or **rejected transparently** during compilation.

---

## Project Status

SparseFlow is in **active research and development**.

Current state:
- ✅ CPU correctness validated
- ✅ Initial GPU functional validation completed
- ⚠️ GPU kernels are not yet performance-optimized
- 🚧 Benchmarking and cross-vendor validation in progress

> SparseFlow is **not** a production-ready framework yet.  
> Claims are intentionally conservative and evidence-driven.

---

## Repository Structure

```text
SparseFlow/
├── compiler/
│   ├── passes/        # MLIR transformation passes
│   ├── runtime/       # CPU/GPU runtime components
│   └── tools/         # Build and test utilities
├── tests/             # Functional and correctness tests
├── benchmarks/        # Early, non-optimized benchmarks
└── docs/              # Design notes and documentation
