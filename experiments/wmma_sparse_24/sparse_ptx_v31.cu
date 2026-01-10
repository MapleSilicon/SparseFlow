#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void sparse_mma_m16n8k32_kernel(
    const half* __restrict__ A,      // Compressed A [M, K/2]
    const half* __restrict__ B,      // B [K, N]
    float* __restrict__ C,           // C [M, N]
    const uint32_t* __restrict__ meta, // Metadata [M, K/16]
    int M, int N, int K) 
{
    // Simplified indexing for a single tile benchmark
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    // Register fragments
    uint32_t ra[4]; // Matrix A: 4 regs for m16n8k32 (32 halfs)
    uint32_t rb[2]; // Matrix B: 2 regs for m16n8k32 (16 halfs)
    float rc[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // Accumulators
    uint32_t rm;    // Metadata: 1 reg

    // Pointer arithmetic (for 16x8x32 tile)
    const uint32_t* a_ptr = (const uint32_t*)A;
    const uint32_t* b_ptr = (const uint32_t*)B;

    // Load Fragments
    ra[0] = a_ptr[0]; ra[1] = a_ptr[1]; ra[2] = a_ptr[2]; ra[3] = a_ptr[3];
    rb[0] = b_ptr[0]; rb[1] = b_ptr[1];
    rm    = meta[0];

    // The PTX Instruction - Explicitly matching m16n8k32 vector sizes
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.f32.f16.f16.f32 "
        "{ %0, %1, %2, %3 }, "
        "{ %4, %5, %6, %7 }, "
        "{ %8, %9 }, "
        "{ %10, %11, %12, %13 }, "
        "%14, 0x0;\n"
        : "=f"(rc[0]), "=f"(rc[1]), "=f"(rc[2]), "=f"(rc[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
          "r"(rb[0]), "r"(rb[1]),
          "f"(rc[0]), "f"(rc[1]), "f"(rc[2]), "f"(rc[3]),
          "r"(rm)
    );

    // Store back to C (simplified for benchmark)
    int row = (warp_id * 16) + (lane_id / 4);
    int col = (lane_id % 4) * 2;
    if (row < M && col < N) {
        C[row * N + col] = rc[0];
        C[row * N + col + 1] = rc[1];
    }
}

extern "C" void launch_sparse_ptx_v31(
    const half* A, const half* B, float* C,
    const uint32_t* meta, int M, int N, int K
) {
    dim3 grid(1, 1);   // Single block for initial test
    dim3 block(256);   // 8 warps
    sparse_mma_m16n8k32_kernel<<<grid, block>>>(A, B, C, meta, M, N, K);
}
