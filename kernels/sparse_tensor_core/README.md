# Sparse Tensor Core Kernels

## Current Status
- `sparse_tc_m16n8k32.cu`: Baseline single-warp tensor core kernel
- Uses mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
- Metadata format: E[g*M + row] (validated in sparseflow-ref-v3)

## Next Steps
1. Validate correctness vs reference
2. Add shared memory for B tiles
3. Multi-warp blocks (2-4 warps)
4. K-loop unrolling
5. Prefetching

## Target Performance
- 512×512: >5 TFLOPS
- 1024×1024: >10 TFLOPS
- 2048×2048: >20 TFLOPS
