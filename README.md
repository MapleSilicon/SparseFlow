# SparseFlow — Phase-2A (Runtime)

Phase-2A delivers a production-grade GPU structured sparsity runtime foundation using cuSPARSELt:
- Correct cuSPARSELt plan lifecycle management
- Content-based compressed caching (no pointer-identity bugs)
- Stable execution on 1024³ and 2048³
- Production-ready error handling

## Quick Start

### 1) Environment
You need:
- Linux (Ubuntu recommended)
- CUDA Toolkit (nvcc, libcudart, cuSPARSELt)
- CMake + C++ compiler
- SQLite3

### 2) Build
```bash
./scripts/env_check.sh
./scripts/build.sh
```

### 3) Benchmark
```bash
./scripts/bench.sh
```

## Expected Results (RTX 3090 / SM86)

* **1024³**: Dense ~43.9 TFLOPS, Sparse 2:4 ~28.8 TFLOPS
* **2048³**: Dense ~59.5 TFLOPS, Sparse 2:4 ~52.5 TFLOPS

✅ Plan init succeeds  
✅ Compression succeeds  
✅ Cache stable across runs  
✅ No memory corruption  

## Status

**Phase-2A: COMPLETE**
- Production runtime infrastructure
- Honest performance measurement
- Ready for investor demos
- Ready for IRAP applications

## Repository Layout

* `sparseflow/runtime/` - C++/CUDA runtime, cache, kernels
* `sparseflow/tests/` - Unit tests
* `scripts/` - Build + benchmark automation
* `CMakeLists.txt` - Build configuration

