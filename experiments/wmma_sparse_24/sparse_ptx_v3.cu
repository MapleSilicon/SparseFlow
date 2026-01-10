#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// PTX Sparse GEMM using mma.sp.sync.m16n8k32
// This is the REAL implementation for 2:4 sparsity
__global__ void sparse_gemm_m16n8k32_ptx(
    const half* __restrict__ A_sp,   // Compressed [M, K/2] (2:4 sparse)
    const uint32_t* __restrict__ E,  // Metadata [M, K/32]
    const half* __restrict__ B,      // Dense [K, N]
    float* __restrict__ C,           // Output [M, N] (fp32 accumulator)
    int M, int N, int K
) {
    // Each block handles 64x64 output tile
    // 8 warps per block (256 threads)
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Warp tiling: 2x4 layout of 16x8 tiles
    int warp_row = (warp_id / 4) * 16;  // 0 or 16
    int warp_col = (warp_id % 4) * 8;   // 0, 8, 16, 24
    
    int tile_m = blockIdx.y * 64 + warp_row;
    int tile_n = blockIdx.x * 64 + warp_col;
    
    if (tile_m >= M || tile_n >= N) return;
    
    // Accumulators: 16x8 tile = 4 fp32 registers per thread
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Process K in chunks of 32
    for (int k_base = 0; k_base < K; k_base += 32) {
        // Load A fragment (compressed): 16 fp16 values (8 uint32_t)
        // For m16n8k32: each thread needs 16 fp16 = 8 uint32
        uint32_t a_frag[4];
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = tile_m + (lane_id / 4);
            int col_compressed = (k_base / 2) + (lane_id % 4) * 2 + i;
            
            if (row < M && col_compressed < K/2) {
                const half* a_ptr = A_sp + row * (K/2) + col_compressed;
                a_frag[i] = *reinterpret_cast<const uint32_t*>(a_ptr);
            } else {
                a_frag[i] = 0;
            }
        }
        
        // Load B fragment (dense): 8 fp16 values (4 uint32_t)
        uint32_t b_frag[2];
        
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int row = k_base + (lane_id / 4) * 2 + i;
            int col = tile_n + (lane_id % 4);
            
            if (row < K && col < N) {
                const half* b_ptr = B + row * N + col;
                b_frag[i] = *reinterpret_cast<const uint32_t*>(b_ptr);
            } else {
                b_frag[i] = 0;
            }
        }
        
        // Load metadata (1 uint32 per 32 K-elements per row)
        uint32_t meta = 0;
        int meta_row = tile_m + (lane_id / 4);
        int meta_col = k_base / 32;
        if (meta_row < M && meta_col < (K/32)) {
            meta = E[meta_row * (K/32) + meta_col];
        }
        
        // Execute mma.sp.sync.m16n8k32
        // Output: 4 fp32 registers per thread (16x8 tile distributed)
        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, "           // d[0:3] - output accumulators
            "{%4, %5, %6, %7}, "           // a[0:3] - compressed matrix A (4x uint32)
            "{%8, %9}, "                   // b[0:1] - dense matrix B (2x uint32)
            "{%10, %11, %12, %13}, "       // c[0:3] - input accumulators
            "%14, 0x0;\n"                  // metadata, selector
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]),
              "r"(meta)
        );
    }
    
    // Store results: distribute 16x8 tile across threads
    // Each thread owns 4 fp32 values
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int out_row = tile_m + (lane_id / 4);
        int out_col = tile_n + (lane_id % 4) + i * 4;
        
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = acc[i];
        }
    }
}

extern "C" void launch_sparse_ptx_v3(
    const half* A_sp, const uint32_t* E, const half* B,
    float* C, int M, int N, int K
) {
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    dim3 block(256);  // 8 warps
    
    sparse_gemm_m16n8k32_ptx<<<grid, block>>>(A_sp, E, B, C, M, N, K);
}
