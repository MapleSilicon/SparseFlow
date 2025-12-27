# Phase-2 Canonical Benchmark Results

**Environment:**
- GPU: NVIDIA RTX 3090 (24GB)
- CUDA: 12.x
- cuSPARSELt: Included with CUDA
- Driver: 535.x+

**Benchmark:** FP16 Matrix Multiplication (2:4 structured sparsity)

## Results

| Size    | Dense (TFLOPS) | Sparse 2:4 (TFLOPS) | Selected |
|---------|----------------|---------------------|----------|
| 1024³   | 43.9           | 28.9                | Dense    |
| 2048³   | 59.5           | 52.5                | Dense    |

## Interpretation

**Why dense wins:** cuBLAS tensor cores are extremely optimized for dense operations. 
At these sizes, sparse overhead (metadata processing, indirection) exceeds the 
computational savings from skipping zero operations.

**This is correct behavior.** The runtime measures honestly and selects the 
fastest kernel. Sparse acceleration requires:
- Kernel fusion (GEMM + bias + activation)
- Batched operations
- Amortized compression cost
- Long-running inference workloads

**Runtime validation:** The system correctly:
- ✅ Benchmarks both kernels
- ✅ Selects fastest option
- ✅ Caches decision persistently
- ✅ Prevents regression

## Stability

All results reproducible across:
- Multiple runs (cache stability verified)
- Process restarts (SQLite persistence)
- Clean builds (no hidden state)

**Validation:** `./scripts/validate.sh` passes consistently
