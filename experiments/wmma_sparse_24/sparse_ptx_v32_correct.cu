#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void sparse_mma_m16n8k32_correct(
    const half* __restrict__ A,      // Dense A [M, K]
    const half* __restrict__ B,      // Sparse B [K, N] (2:4 packed)
    float* __restrict__ C,           // Output [M, N]
    const uint32_t* __restrict__ meta, // Metadata [single u32 per tile]
    int M, int N, int K) 
{
    int lane_id = threadIdx.x & 31;
    
    // Load A: 4 × u32 registers (8 halfs)
    uint32_t a0, a1, a2, a3;
    const uint32_t* a_ptr = (const uint32_t*)A;
    a0 = a_ptr[0];
    a1 = a_ptr[1];
    a2 = a_ptr[2];
    a3 = a_ptr[3];
    
    // Load B: 2 × u32 registers (4 halfs, sparse packed)
    uint32_t b0, b1;
    const uint32_t* b_ptr = (const uint32_t*)B;
    b0 = b_ptr[0];
    b1 = b_ptr[1];
    
    // Load metadata: 1 × u32 register
    uint32_t e = meta[0];
    
    // Accumulators: 4 × f32 registers
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    
    // THE ONLY CORRECT SPARSE MMA INSTRUCTION
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "      // accumulator OUT
        "{%4,%5,%6,%7}, "      // A matrix (4 u32)
        "{%8,%9}, "            // B matrix (2 u32, sparse)
        "%10, "                // metadata (1 u32)
        "{%0,%1,%2,%3};"       // accumulator IN
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(e)
    );
    
    // Store results
    int row = lane_id / 4;
    int col = (lane_id % 4) * 2;
    if (row < M && col < N) {
        C[row * N + col] = d0;
        C[row * N + col + 1] = d1;
    }
}

extern "C" void launch_sparse_mma_correct(
    const half* A, const half* B, float* C,
    const uint32_t* meta, int M, int N, int K
) {
    dim3 grid(1, 1);
    dim3 block(32);  // Single warp test
    sparse_mma_m16n8k32_correct<<<grid, block>>>(A, B, C, meta, M, N, K);
}
