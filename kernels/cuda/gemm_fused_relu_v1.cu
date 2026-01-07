#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 16

__global__ void gemm_fused_relu_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half smem_A[BM][BK];
    __shared__ half smem_B[BK][BN];
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 64;
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    for (int i = 0; i < 8; i++) {
        wmma::fill_fragment(acc[i], 0.0f);
    }
    
    for (int k = 0; k < K; k += BK) {
        for (int i = tid; i < BM * BK; i += 256) {
            int row = i / BK;
            int col = i % BK;
            int global_row = block_row + row;
            int global_col = k + col;
            smem_A[row][col] = (global_row < M && global_col < K) ? 
                A[global_row * K + global_col] : __float2half(0.0f);
        }
        
        for (int i = tid; i < BK * BN; i += 256) {
            int row = i / BN;
            int col = i % BN;
            int global_row = k + row;
            int global_col = block_col + col;
            smem_B[row][col] = (global_row < K && global_col < N) ? 
                B[global_row * N + global_col] : __float2half(0.0f);
        }
        
        __syncthreads();
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
        
        wmma::load_matrix_sync(a_frag[0], &smem_A[warp_row][0], BK);
        wmma::load_matrix_sync(a_frag[1], &smem_A[warp_row + 16][0], BK);
        wmma::load_matrix_sync(b_frag[0], &smem_B[0][warp_col], BN);
        wmma::load_matrix_sync(b_frag[1], &smem_B[0][warp_col + 16], BN);
        wmma::load_matrix_sync(b_frag[2], &smem_B[0][warp_col + 32], BN);
        wmma::load_matrix_sync(b_frag[3], &smem_B[0][warp_col + 48], BN);
        
        wmma::mma_sync(acc[0], a_frag[0], b_frag[0], acc[0]);
        wmma::mma_sync(acc[1], a_frag[0], b_frag[1], acc[1]);
        wmma::mma_sync(acc[2], a_frag[0], b_frag[2], acc[2]);
        wmma::mma_sync(acc[3], a_frag[0], b_frag[3], acc[3]);
        wmma::mma_sync(acc[4], a_frag[1], b_frag[0], acc[4]);
        wmma::mma_sync(acc[5], a_frag[1], b_frag[1], acc[5]);
        wmma::mma_sync(acc[6], a_frag[1], b_frag[2], acc[6]);
        wmma::mma_sync(acc[7], a_frag[1], b_frag[3], acc[7]);
        
        __syncthreads();
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < acc[i].num_elements; j++) {
            acc[i].x[j] = fmaxf(acc[i].x[j], 0.0f);
        }
    }
    
    int out_row = block_row + warp_row;
    int out_col = block_col + warp_col;
    
    if (out_row + 31 < M && out_col + 63 < N) {
        wmma::store_matrix_sync(&C[out_row * N + out_col], acc[0], N, wmma::mem_row_major);
        wmma::store_matrix_sync(&C[out_row * N + out_col + 16], acc[1], N, wmma::mem_row_major);
        wmma::store_matrix_sync(&C[out_row * N + out_col + 32], acc[2], N, wmma::mem_row_major);
        wmma::store_matrix_sync(&C[out_row * N + out_col + 48], acc[3], N, wmma::mem_row_major);
        wmma::store_matrix_sync(&C[(out_row + 16) * N + out_col], acc[4], N, wmma::mem_row_major);
        wmma::store_matrix_sync(&C[(out_row + 16) * N + out_col + 16], acc[5], N, wmma::mem_row_major);
        wmma::store_matrix_sync(&C[(out_row + 16) * N + out_col + 32], acc[6], N, wmma::mem_row_major);
        wmma::store_matrix_sync(&C[(out_row + 16) * N + out_col + 48], acc[7], N, wmma::mem_row_major);
    }
}

extern "C" void launch_gemm_fused_relu(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_fused_relu_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
