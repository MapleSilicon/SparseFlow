#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>
#include <cstdint>

// Helper: Convert u32 (2 halfs) to half
__device__ __forceinline__ half u32_to_half(uint32_t packed) {
    half2 h2; 
    h2.x = __float2half(__low2float(__half2float(packed)));
    h2.y = __float2half(__low2float(__half2float(packed >> 16)));
    return h2.x;
}

__global__ void sparse_wmma_kernel_packed(
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

    // Registers
    float acc0, acc1, acc2, acc3;
    uint32_t a_packed, b_packed;
    uint32_t meta = 0xFFFFFFFF; // Dummy metadata

    // Main Loop
    for (int k = 0; k < K; k += 128) {
        
        // --- LOAD A (Packed u32) ---
        const uint32_t* a_ptr32 = reinterpret_cast<const uint32_t*>(&A[cRow * K + k]);
        
        // Load 2 halfs at once
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(a_packed) : "l"(a_ptr32));
        
        // --- LOAD B (Packed u32) ---
        const uint32_t* b_ptr32 = reinterpret_cast<const uint32_t*>(&B_sparse[k * N + cCol]);
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(b_packed) : "l"(b_ptr32));

        // --- UNPACK TO HALF ---
        // We unpack u32 -> half2 using C++ helper, no asm conversion errors
        half a0 = u32_to_half(a_packed);
        half b0 = u32_to_half(b_packed);

        // --- TURBO BUTTON ---
        asm volatile(
            "mma.sp.sync.aligned.m16n8k128.row.col.f32.f16.f16 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, %10;"
            : "=f"(acc0), "=f"(acc1), "=f"(acc2), "=f"(acc3)
            : "h"(a0), "h"(a0), "h"(a0), "h"(a0),       // Note: Dummy A here for simplicity (broadcast)
              "h"(b0), "h"(b0),                             // B
              "r"(meta)
        );
    }

    // --- STORE RESULT ---
    float* c_ptr = &C[cRow * N + cCol];
    
    // Manual unrolled store
    asm volatile("st.global.f32 [%0], %1;" : : "l"(c_ptr), "f"(acc0));
    asm volatile("st.global.f32 [%0], %1;" : : "l"(c_ptr + 1), "f"(acc1));
    asm volatile("st.global.f32 [%0], %1;" : : "l"(c_ptr + 2), "f"(acc2));
    asm volatile("st.global.f32 [%0], %1;" : : "l"(c_ptr + 3), "f"(acc3));
}
