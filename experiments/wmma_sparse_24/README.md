# 2:4 Structured Sparse WMMA

Implementation of NVIDIA's 2:4 structured sparsity using `mma.sp.sync` PTX instruction.

## Features
- **2× Theoretical Speedup**: Exploits 2:4 sparsity pattern
- **50% Memory Reduction**: Compressed weight storage
- **Native Hardware Support**: Uses Ampere Tensor Cores

## Build & Run
```bash
make
make run
```

## Expected Results
- Dense baseline: ~125 TFLOPS
- Sparse 2:4: ~200-240 TFLOPS
- **Target speedup: 1.8-2.0×**

## Technical Details
- PTX instruction: `mma.sp.sync.aligned.m16n8k16`
- Metadata: 2 bits per element (packed in uint32_t)
- Pattern: Top-2 magnitude selection per 4-element group
