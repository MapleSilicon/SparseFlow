#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// 8 warps/block (256 threads). Block tile = 128x128 (2x4 warps of 16x16 tiles).
__global__ void dense_wmma_kernel_fp32_out(
    const half* __restrict__ A,        // row-major [M,K]
    const half* __restrict__ B_col,    // col-major view of [K,N] => pass B.t().contiguous()
    float* __restrict__ C,             // row-major [M,N] fp32
    int M, int N, int K
) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;          // 0..7
    int warp_row = (warp_id / 4);    // 0..1
    int warp_col = (warp_id % 4);    // 0..3

    int tile_m = (blockIdx.y * 128) + warp_row * 16;
    int tile_n = (blockIdx.x * 128) + warp_col * 16;
    if (tile_m >= M || tile_n >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;

    wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const half* A_tile = A + tile_m * K + k0;
        const half* B_tile = B_col + tile_n * K + k0; // B_col is N x K (because B.t() is N x K)
        wmma::load_matrix_sync(a_frag, A_tile, K);
        wmma::load_matrix_sync(b_frag, B_tile, K);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    float* C_tile = C + tile_m * N + tile_n;
    wmma::store_matrix_sync(C_tile, acc, N, wmma::mem_row_major);
}

extern "C" void launch_dense_wmma_fp32(
    const half* A, const half* B_col, float* C,
    int M, int N, int K
) {
    dim3 block(256);
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    dense_wmma_kernel_fp32_out<<<grid, block>>>(A, B_col, C, M, N, K);
}
