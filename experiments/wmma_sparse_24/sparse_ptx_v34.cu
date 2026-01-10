#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda::wmma;

__global__ void sparse_wmma_kernel_ptx_v34(
    const half* __restrict__ A, 
    const half* __restrict__ B_sparse, 
    float* __restrict__ C, 
    int M, int N, int K
) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    if (warp_id >= 16) return; // 16 warps for 128x128 block

    int block_row = blockIdx.y * 128;
    int block_col = blockIdx.x * 128;

    int warp_row = (warp_id / 8) * 16; 
    int warp_col = (warp_id % 8) * 8;  

    int cRow = block_row + warp_row;
    int cCol = block_col + warp_col;

    if (cRow >= M || cCol >= N) return;

    fragment<matrix_a, 16, 8, 128, half, row_major> a_frag;
    fragment<matrix_b, 16, 8, 128, half, col_major> b_frag;
    fragment<accumulator, 16, 8, 128, float> acc_frag;

    fill_fragment(acc_frag, 0.0f);

    half a_regs[4];
    half b_regs[2];
    float acc_regs[4];

    // Main Loop
    for (int k = 0; k < K; k += 128) {
        // Load
        load_matrix_sync(a_frag, &A[cRow * K + k], K);
        load_matrix_sync(b_frag, &B_sparse[k * N + cCol], N);

        // Unpack
        #pragma unroll
        for (int i = 0; i < 4; i++) a_regs[i] = a_frag.x[i];
        #pragma unroll
        for (int i = 0; i < 2; i++) b_regs[i] = b_frag.x[i];
        #pragma unroll
        for (int i = 0; i < 4; i++) acc_regs[i] = acc_frag.x[i];

        // TURBO BUTTON
        uint32_t meta = 0xFFFFFFFF;

        asm volatile(
            "mma.sp.sync.aligned.m16n8k128.row.col.f32.f16.f16 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, %10;"
            : "=f"(acc_regs[0]), "=f"(acc_regs[1]), "=f"(acc_regs[2]), "=f"(acc_regs[3])
            : "h"(a_regs[0]), "h"(a_regs[1]), "h"(a_regs[2]), "h"(a_regs[3]),
              "h"(b_regs[0]), "h"(b_regs[1]), 
              "r"(meta)
        );

        // Pack back
        #pragma unroll
        for (int i = 0; i < 4; i++) acc_frag.x[i] = acc_regs[i];
    }

    store_matrix_sync(C + cRow * N + cCol, acc_frag, N, mem_row_major);
}
