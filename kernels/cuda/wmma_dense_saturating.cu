#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128  // Block tile M
#define BN 128  // Block tile N
#define BK 32   // Block tile K

#define WARPS_M 4  // 4 warps in M dimension (each warp handles 32 rows via 2x16x16 tiles)
#define WARPS_N 2  // 2 warps in N dimension (each warp handles 32 cols via 2x16x16 tiles)
#define THREADS (WARPS_M * WARPS_N * 32)  // 256 threads

__global__ void __launch_bounds__(THREADS)
wmma_dense_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tile
    __shared__ half smem_A[BM][BK];
    __shared__ half smem_B[BK][BN];
    
    // Thread and warp IDs
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp position in block
    int warp_row = (warp_id / WARPS_N) * 2;  // Each warp handles 2x 16x16 tiles vertically
    int warp_col = (warp_id % WARPS_N) * 2;  // Each warp handles 2x 16x16 tiles horizontally
    
    // Block position in grid
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    
    // Accumulators (4 tiles per warp: 2x2)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[4];
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::fill_fragment(acc[i], __float2half(0.0f));
    }
    
    // Loop over K dimension
    for (int k_base = 0; k_base < K; k_base += BK) {
        // Load tile of A into shared memory (cooperative loading)
        #pragma unroll
        for (int i = tid; i < BM * BK; i += THREADS) {
            int row = i / BK;
            int col = i % BK;
            int global_row = block_row + row;
            int global_col = k_base + col;
            
            smem_A[row][col] = (global_row < M && global_col < K) 
                ? A[global_row * K + global_col] 
                : __float2half(0.0f);
        }
        
        // Load tile of B into shared memory
        #pragma unroll
        for (int i = tid; i < BK * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int global_row = k_base + row;
            int global_col = block_col + col;
            
            smem_B[row][col] = (global_row < K && global_col < N)
                ? B[global_row * N + global_col]
                : __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute 2x2 tiles per warp
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];
        
        // Iterate over BK in chunks of WMMA_K
        #pragma unroll
        for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
            // Load A fragments (2 vertical tiles)
            wmma::load_matrix_sync(a_frag[0], &smem_A[warp_row * WMMA_M][k_step], BK);
            wmma::load_matrix_sync(a_frag[1], &smem_A[(warp_row + 1) * WMMA_M][k_step], BK);
            
            // Load B fragments (2 horizontal tiles)
            wmma::load_matrix_sync(b_frag[0], &smem_B[k_step][warp_col * WMMA_N], BN);
            wmma::load_matrix_sync(b_frag[1], &smem_B[k_step][(warp_col + 1) * WMMA_N], BN);
            
            // Compute 2x2 tile matrix multiply
            wmma::mma_sync(acc[0], a_frag[0], b_frag[0], acc[0]);  // top-left
            wmma::mma_sync(acc[1], a_frag[0], b_frag[1], acc[1]);  // top-right
            wmma::mma_sync(acc[2], a_frag[1], b_frag[0], acc[2]);  // bottom-left
            wmma::mma_sync(acc[3], a_frag[1], b_frag[1], acc[3]);  // bottom-right
        }
        
        __syncthreads();
    }
    
    // Store results
    int out_row = block_row + warp_row * WMMA_M;
    int out_col = block_col + warp_col * WMMA_N;
    
    if (out_row < M && out_col < N) {
        wmma::store_matrix_sync(&C[out_row * N + out_col], acc[0], N, wmma::mem_row_major);
    }
    if (out_row < M && out_col + WMMA_N < N) {
        wmma::store_matrix_sync(&C[out_row * N + out_col + WMMA_N], acc[1], N, wmma::mem_row_major);
    }
    if (out_row + WMMA_M < M && out_col < N) {
        wmma::store_matrix_sync(&C[(out_row + WMMA_M) * N + out_col], acc[2], N, wmma::mem_row_major);
    }
    if (out_row + WMMA_M < M && out_col + WMMA_N < N) {
        wmma::store_matrix_sync(&C[(out_row + WMMA_M) * N + out_col + WMMA_N], acc[3], N, wmma::mem_row_major);
    }
}

extern "C" void launch_wmma_dense(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS);
    
    wmma_dense_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
