# SparseFlow Phase-2: Canonical Benchmark Results

**Hardware:** NVIDIA RTX 3090 (24GB, SM86)  
**CUDA:** 12.x  
**cuSPARSELt:** Included with CUDA Toolkit  
**Date:** December 2025  
**Version:** v0.2.0

---

## FP16 Matrix Multiplication (2:4 Structured Sparsity)

| Matrix Size | Dense (TFLOPS) | Sparse 2:4 (TFLOPS) | Runtime Selected |
|-------------|----------------|---------------------|------------------|
| 1024³       | 43.9           | 28.9                | Dense            |
| 2048³       | 59.5           | 52.5                | Dense            |

**Runtime Decision:** The system correctly selects dense when it outperforms sparse.

**Reproducibility:** `./scripts/validate.sh` (exits 0 on success)

**Repository:** https://github.com/MapleSilicon/SparseFlow (tag: v0.2.0)
