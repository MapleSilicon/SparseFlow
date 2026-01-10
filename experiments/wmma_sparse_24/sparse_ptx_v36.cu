#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.hpp>

__global__ void sparse_wmma_kernel_pure_ptx(
    const half* __restrict__ A, 
    const half* __restrict__ B_sparse, 
    float* __restrict__ C, 
    int M, int N, int K
) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    if (warp_id >= 16) return; 

    int block_row = blockIdx.y * 128;
    int block_col = blockIdx.x * 128;

    int warp_row = (warp_id / 8) * 16; 
    int warp_col = (warp_id % 8) * 8;  

    int cRow = block_row + warp_row;
    int cCol = block_col + warp_col;

    if (cRow >= M || cCol >= N) return;

    // 1. Registers
    float acc0, acc1, acc2, acc3;
    
    half a0, a1, a2, a3;
    half b0, b1;

    // Init Accumulator
    acc0 = 0.0f; acc1 = 0.0f;
    acc2 = 0.0f; acc3 = 0.0f;

    // 2. Main Loop
    for (int k = 0; k < K; k += 128) {
        
        // --- LOAD A (Fix: Use Address Register) ---
        const half* a_ptr = &A[cRow * K + k];
        
        // Calculate addresses in registers to avoid "lvalue" error
        uint32_t addr_a0 = (uint32_t)(a_ptr + 0);
        uint32_t addr_a1 = (uint32_t)(a_ptr + 8);
        uint32_t addr_a2 = (uint32_t)(a_ptr + 16);
        uint32_t addr_a3 = (uint32_t)(a_ptr + 24);

        asm volatile("ld.global.nc.f16 %0, [%1];" : "=h"(a0) : "l"(addr_a0));
        asm volatile("ld.global.nc.f16 %0, [%1];" : "=h"(a1) : "l"(addr_a1));
        asm volatile("ld.global.nc.f16 %0, [%1];" : "=h"(a2) : "l"(addr_a2));
        asm volatile("ld.global.nc.f16 %0, [%1];" : "=h"(a3) : "l"(addr_a3));

        // --- LOAD B ---
        const half* b_ptr = &B_sparse[k * N + cCol];
        uint32_t addr_b0 = (uint32_t)(b_ptr + 0);
        uint32_t addr_b1 = (uint32_t)(b_ptr + 8);

        asm volatile("ld.global.nc.f16 %0, [%1];" : "=h"(b0) : "l"(addr_b0));
        asm volatile("ld.global.nc.f16 %0, [%1];" : "=h"(b1) : "l"(addr_b1));

        // --- TURBO BUTTON (Sparse MMA) ---
        // Instruction: m16n8k128.row.col
        uint32_t meta = 0xFFFFFFFF; // Dummy metadata

        asm volatile(
            "mma.sp.sync.aligned.m16n8k128.row.col.f32.f16.f16 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, %10;"
            : "=f"(acc0), "=f"(acc1), "=f"(acc2), "=f"(acc3)
            : "h"(a0), "h"(a1), "h"(a2), "h"(a3),
              "h"(b0), "h"(b1), 
              "r"(meta)
        );
    }

    // 3. STORE RESULT
    float* c_ptr = &C[cRow * N + cCol];
    
    uint32_t addr_c0 = (uint32_t)(c_ptr + 0);
    uint32_t addr_c1 = (uint32_t)(c_ptr + 1);
    uint32_t addr_c2 = (uint32_t)(c_ptr + 2);
    uint32_t addr_c3 = (uint32_t)(c_ptr + 3);

    asm volatile("st.global.f32 [%0], %1;" : : "l"(addr_c0), "f"(acc0));
    asm volatile("st.global.f32 [%0], %1;" : : "l"(addr_c1), "f"(acc1));
    asm volatile("st.global.f32 [%0], %1;" : : "l"(addr_c2), "f"(acc2));
    asm volatile("st.global.f32 [%0], %1;" : : "l"(addr_c3), "f"(acc3));
}
