# SparseFlow GPU Kernel ABI (v0.3-alpha)

This document freezes the *experimental* ABI used by SparseFlow's GPU rewrite path.
The goal of v0.3-alpha is **IR plumbing correctness**, not performance.

## Scope
v0.3-alpha guarantees:
- `sparseflow-gpu-rewrite` inserts a valid `gpu.module`, `gpu.func` (kernel stub), and `gpu.launch`.
- The pass pipeline can run end-to-end without crashes/verifier errors.

v0.3-alpha does **not** guarantee:
- Correct numeric results on GPU
- Any GPU performance claims
- Final kernel lowering to NVVM/ROCDL/SPIR-V

## ABI: Kernel Signature (planned target)
The intended kernel ABI for SparseFlow N:M matmul is:
```mlir
gpu.func @sparseflow_nm_kernel(
  %A       : memref<?x?xf32>,   // Sparse matrix A (structured N:M)
  %B       : memref<?x?xf32>,   // Dense matrix B
  %C       : memref<?x?xf32>,   // Output matrix
  %rowMask : memref<?xi32>,     // Row-wise mask (format TBD)
  %M       : i32,               // Rows of C
  %N       : i32,               // Cols of C
  %K       : i32,               // Inner dimension
  %n       : i32,               // N in N:M
  %m       : i32                // M in N:M
) kernel
```

### Notes
- v0.3-alpha currently emits a **stub kernel** (`gpu.func @kernel()`) to validate GPU dialect wiring.
- The kernel ABI above is the **next step** once bufferization and call-consumption are fully stabilized.

## Launch Mapping (conceptual)
- `blockIdx.x / blockIdx.y` map to output tiles
- `threadIdx.x / threadIdx.y` map to elements inside a tile

Tiling constants are intentionally **not frozen** in v0.3-alpha.
