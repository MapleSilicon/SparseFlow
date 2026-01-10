#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Inline PTX for mma.sp.sync (2:4 structured sparse)
__device__ __forceinline__ void mma_sp_sync_m16n8k16(
    float* d, const uint32_t* a, const uint32_t* b, 
    const float* c, uint32_t e, int trans_a, int trans_b
) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "%14, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e)
    );
}

// 2:4 Sparse WMMA kernel
// Block: 128x128 tile, 8 warps (256 threads)
// Each warp: 16x16 tile using mma.sp.sync
__global__ void sparse_24_kernel(
    const half* __restrict__ A,      // Dense [M, K] row-major
    const half* __restrict__ Bc,     // Compressed [K, N/2] - 50% size
    const uint32_t* __restrict__ E,  // Metadata [K/16, N/8] 
    half* __restrict__ C,            // Output [M, N]
    int M, int N, int K
) {
    // Warp and thread IDs
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp position in 128x128 block tile
    int warp_row = (warp_id / 4) * 16;  // 0 or 16
    int warp_col = (warp_id % 4) * 32;  // 0, 32, 64, 96
    
    // Global tile position
    int tile_m = blockIdx.y * 128 + warp_row;
    int tile_n = blockIdx.x * 128 + warp_col;
    
    if (tile_m >= M || tile_n >= N) return;
    
    // Accumulators for 16x32 output (2x mma.sp which produces 16x8 each)
    float acc[8] = {0.0f};
    
    // Process K dimension in chunks of 16
    for (int k = 0; k < K; k += 16) {
        // Load A fragment (16x16 from dense matrix)
        uint32_t a_frag[4];
        for (int i = 0; i < 4; i++) {
            int row = tile_m + lane_id / 4;
            int col = k + (lane_id % 4) * 4 + i;
            if (row < M && col < K) {
                const half* a_ptr = A + row * K + col;
                a_frag[i] = *reinterpret_cast<const uint32_t*>(a_ptr);
            } else {
                a_frag[i] = 0;
            }
        }
        
        // Load Bc fragments (2x 16x8 from compressed matrix)
        // Bc is stored compressed: for each 16x16 block, only 16x8 elements
        uint32_t bc_frag0[2], bc_frag1[2];
        int bc_row = k;
        int bc_col0 = tile_n / 2;      // First 16x8 tile
        int bc_col1 = (tile_n + 16) / 2; // Second 16x8 tile
        
        for (int i = 0; i < 2; i++) {
            int row = bc_row + lane_id / 4;
            int col0 = bc_col0 + (lane_id % 4) * 2 + i;
            int col1 = bc_col1 + (lane_id % 4) * 2 + i;
            
            if (row < K) {
                bc_frag0[i] = *reinterpret_cast<const uint32_t*>(&Bc[row * (N/2) + col0]);
                bc_frag1[i] = *reinterpret_cast<const uint32_t*>(&Bc[row * (N/2) + col1]);
            } else {
                bc_frag0[i] = 0;
                bc_frag1[i] = 0;
            }
        }
        
        // Load metadata (2 bits per element, packed in uint32_t)
        int meta_row = k / 16;
        int meta_col0 = tile_n / 8;
        int meta_col1 = (tile_n + 16) / 8;
        uint32_t meta0 = E[meta_row * (N/8) + meta_col0];
        uint32_t meta1 = E[meta_row * (N/8) + meta_col1];
        
        // Perform mma.sp.sync for two 16x8 tiles
        float tmp0[4], tmp1[4];
        
        mma_sp_sync_m16n8k16(tmp0, a_frag, bc_frag0, &acc[0], meta0, 0, 0);
        mma_sp_sync_m16n8k16(tmp1, a_frag, bc_frag1, &acc[4], meta1, 0, 0);
        
        // Accumulate results
        for (int i = 0; i < 4; i++) {
            acc[i] += tmp0[i];
            acc[i + 4] += tmp1[i];
        }
    }
    
    // Store results (convert fp32 -> fp16)
    for (int i = 0; i < 8; i++) {
        int out_row = tile_m + lane_id / 8;
        int out_col = tile_n + (lane_id % 8) * 4 + i;
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = __float2half(acc[i]);
        }
    }
}

extern "C" void launch_sparse_24_wmma(
    const half* A, const half* Bc, const uint32_t* E,
    half* C, int M, int N, int K
) {
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    dim3 block(256);
    sparse_24_kernel<<<grid, block>>>(A, Bc, E, C, M, N, K);
}
