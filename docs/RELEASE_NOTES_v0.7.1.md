# SparseFlow v0.7.1 – SPA + CPU Runtime + Python CLI (Dev Preview)

This release packages the current SparseFlow stack into a usable, reproducible demo
for researchers, compiler engineers, and hardware partners.

## Highlights

- **SPA v0.6 (2D Sparsity)**
  - Propagates row + column sparsity through a small MLIR pipeline
  - Exports JSON metadata with row/col masks and sparsity stats
  - Proven on structured sparse matmuls with 50% row + 50% col sparsity (≈75% FLOP reduction)

- **C++ CPU Runtime (OpenMP)**
  - Consumes SPA JSON masks and skips zero rows/cols
  - Blocked matmul kernel with OpenMP parallelism
  - Achieves **≈3–5× speedup** on larger matmuls (512–1024) at ~75% sparsity
  - Honest behavior: small sizes (<512) can be 1–3× due to OpenMP and cache overhead

- **End-to-End Demo Script**
  - `./spa-runner.sh`:
    - Rebuilds passes/runtime if needed
    - Runs SPA on test MLIR
    - Exports `spa_sparsity.json`
    - Benchmarks dense vs sparse CPU matmuls

- **Python CLI (Developer Preview)**
  Install:

      cd sparseflow_package
      pip install -e .

  Tools:

      sparseflow-demo
      sparseflow-analyze tests/test_spa_v6_full_2d.mlir
      sparseflow-benchmark

## Quick Start

```bash
git clone https://github.com/MapleSilicon/SparseFlow.git
cd SparseFlow
./spa-runner.sh

