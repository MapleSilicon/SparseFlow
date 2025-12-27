# Phase-3 Week-1 Results (Dense vs Dense-"Fused")

**Date:** December 2025  
**Goal:** Establish credible baseline for dense GEMM and a first fused pipeline (GEMM + bias+ReLU) to measure overhead.

## Hardware / Software
- **GPU:** NVIDIA RTX 3090 (SM86, 24GB)
- **CUDA:** 12.x
- **cuBLAS:** CUDA Toolkit
- **Notes:** Dense-"Fused" currently = GEMM + separate bias/ReLU CUDA kernel (not true epilogue fusion)

## Results

| Size | Dense Unfused TFLOPS | Dense Unfused ms | Dense "Fused" TFLOPS | Dense "Fused" ms |
|------|----------------------|------------------|----------------------|------------------|
| 1024³ | 83.8288 | 0.0256175 | 66.0683 | 0.032504 |
| 2048³ | 105.89  | 0.162243  | 90.3357 | 0.190178 |

## Interpretation (honest)
- The current fused path is slower because it adds an extra kernel + memory traffic after GEMM.
- This baseline is still valuable: it quantifies the fusion overhead and sets targets for real fusion (epilogue fusion / kernel fusion) in Week-2/3.

## Repro
- `./scripts/bench_phase3.sh`
