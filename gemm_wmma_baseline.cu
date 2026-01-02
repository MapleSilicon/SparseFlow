#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 8
#define WMMA_K 16

extern "C" __global__ void gemm_ptx_minimal(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ alignas(16) half smem_A[16][16];
    __shared__ alignas(16) half smem_B[16][8];

    int lane = threadIdx.x & 31;
    int block_row = blockIdx.y * 16;
    int block_col = blockIdx.x * 8;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k_tile = 0; k_tile < K; k_tile += 16) {

        // Load A
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16;
            int c = i % 16;
            smem_A[r][c] =
                (block_row + r < M && k_tile + c < K)
                ? A[(block_row + r) * K + (k_tile + c)]
                : __float2half(0.0f);
        }

        // Load B (row-major)
        for (int i = lane; i < 128; i += 32) {
            int r = i / 8;
            int c = i % 8;
            smem_B[r][c] =
                (k_tile + r < K && block_col + c < N)
                ? B[(k_tile + r) * N + (block_col + c)]
                : __float2half(0.0f);
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, (half*)smem_A, 16);
        wmma::load_matrix_sync(b_frag, (half*)smem_B, 8);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(
        C + block_row * N + block_col,
        acc_frag,
        N,
        wmma::mem_row_major
    );
}

extern "C" void launch_minimal(
    const half* A, const half* B, float* C,
    int M, int N, int K
) {
    dim3 block(32);
    dim3 grid((N + 7) / 8, (M + 15) / 16);
    gemm_ptx_minimal<<<grid, block>>>(A, B, C, M, N, K);
}
