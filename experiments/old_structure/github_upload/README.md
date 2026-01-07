# SparseFlow: MLIR Compiler for N:M Structured Sparsity

## Overview
SparseFlow is a sophisticated MLIR-based compiler targeting 2× sparse acceleration through custom GPU kernel optimization. This repository contains our systematic fusion analysis and optimized WMMA Tensor Core implementations.

## Technical Achievements

### Week 3 Fusion Analysis
- **Systematic evaluation** of existing fusion approaches
- **Measured performance gaps**: -14% (separate kernels) to -89% (untuned libraries)
- **Empirical evidence** justifying custom kernel development

### Optimized WMMA Kernels  
- **22.6 TFLOPS verified performance** on RTX 3090
- **Mathematical correctness** across all matrix sizes
- **Advanced optimizations**: vectorized float4 loading, overflow-safe indexing

## Key Results
```
Matrix Size    | Performance | Verification
256³          | 2.9 TFLOPS  | ✅ PASSED
512³          | 11.5 TFLOPS | ✅ PASSED  
1024³         | 19.0 TFLOPS | ✅ PASSED
2048³         | 22.7 TFLOPS | ✅ PASSED
```

## Repository Structure
- `week3_fusion_analysis/` - Comprehensive fusion benchmarking
- `optimized_kernels/` - Production WMMA Tensor Core kernels
- `compiler_passes/` - MLIR 16×16 tiled GPU rewrite passes
- `benchmarks/` - Performance validation scripts

## Grant Application Evidence
This work demonstrates:
- **Technical depth** through systematic optimization
- **Engineering rigor** with mathematical verification
- **Advanced understanding** of GPU architecture
- **Clear development roadmap** for 40+ TFLOPS optimization

## Next Steps: Phase-3 Development
- Double buffering with cp.async for 40-80 TFLOPS
- Advanced occupancy optimization  
- Production sparse acceleration targeting 2× speedups

## Technical Contact
Gourav - MapleSilicon Founder
Advanced GPU kernel optimization and MLIR compiler development
