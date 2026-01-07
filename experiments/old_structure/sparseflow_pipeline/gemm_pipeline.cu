#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

extern "C" __global__ void gemm_pipeline_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Double-buffered shared memory
    __shared__ half smem_A[2][WMMA_M * WMMA_K];
    __shared__ half smem_B[2][WMMA_K * WMMA_N];

    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;

    int row = tile_m * WMMA_M;
    int col = tile_n * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int tid = threadIdx.x;

    // Preload tile 0
    if (tid < 16 && row + tid < M) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(__cvta_generic_to_shared(&smem_A[0][tid * WMMA_K])),
               "l"(&A[(row + tid) * K])
        );
    }

    if (tid < 16 && col + tid < N) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(__cvta_generic_to_shared(&smem_B[0][tid * WMMA_N])),
               "l"(&B[(col + tid) * K])
        );
    }

    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    for (int k = 0; k < K; k += WMMA_K) {
        int curr = (k / WMMA_K) % 2;
        int next = 1 - curr;

        // Preload next tile
        if (k + WMMA_K < K) {
            if (tid < 16 && row + tid < M) {
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :: "r"(__cvta_generic_to_shared(&smem_A[next][tid * WMMA_K])),
                       "l"(&A[(row + tid) * K + k + WMMA_K])
                );
            }

            if (tid < 16 && col + tid < N) {
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :: "r"(__cvta_generic_to_shared(&smem_B[next][tid * WMMA_N])),
                       "l"(&B[(col + tid) * K + k + WMMA_K])
                );
            }
            asm volatile("cp.async.commit_group;");
        }

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, smem_A[curr], WMMA_K);
        wmma::load_matrix_sync(b_frag, smem_B[curr], WMMA_K);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }

    if (row < M && col < N) {
        wmma::store_matrix_sync(&C[row * N + col], acc, N, wmma::mem_row_major);
    }
}
