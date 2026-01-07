#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 16

extern "C" __global__ void gemm_ptx_scaled(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ alignas(16) half smem_A[BM][BK];  // 128x16
    __shared__ alignas(16) half smem_B[BK][BN];  // 16x128 (row-major)

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // 4x2 warp grid
    const int warp_row = (warp / 2) * 32;
    const int warp_col = (warp % 2) * 64;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[2][8][4];
    #pragma unroll
    for (int mi=0; mi<2; mi++)
      for (int ni=0; ni<8; ni++)
        for (int r=0; r<4; r++)
          acc[mi][ni][r] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {

        // =====================================================
        // DEBUG LOADER: ONLY WARP 0 LOADS SHARED MEMORY
        // =====================================================
        if (warp == 0) {
            // A tile
            for (int i = lane; i < BM*BK; i += 32) {
                int r = i / BK;
                int c = i % BK;
                int gr = block_row + r;
                int gc = k0 + c;
                smem_A[r][c] = (gr < M && gc < K)
                    ? A[gr*K + gc]
                    : __float2half(0.0f);
            }

            // B tile
            for (int i = lane; i < BK*BN; i += 32) {
                int r = i / BN;
                int c = i % BN;
                int gr = k0 + r;
                int gc = block_col + c;
                smem_B[r][c] = (gr < K && gc < N)
                    ? B[gr*N + gc]
                    : __float2half(0.0f);
            }
        }
        __syncthreads();

        // =====================================================
        // COMPUTE
        // =====================================================
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            uint32_t a0,a1,a2,a3;

            uint32_t ptr_a = __cvta_generic_to_shared(
                &smem_A[warp_row + mi*16 + (lane % 16)][(lane / 16) * 8]
            );

            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                : "r"(ptr_a)
            );

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                uint32_t b0,b1;

                uint32_t ptr_b = __cvta_generic_to_shared(
                    &smem_B[lane % 16][warp_col + ni*8]
                );

                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                    : "=r"(b0), "=r"(b1)
                    : "r"(ptr_b)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                      "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1)
                );
            }
        }
        __syncthreads();
    }

    // =====================================================
    // STORE
    // =====================================================
    #pragma unroll
    for (int mi=0; mi<2; mi++) {
      #pragma unroll
      for (int ni=0; ni<8; ni++) {
        int r = block_row + warp_row + mi*16 + (lane / 4);
        int c = block_col + warp_col + ni*8 + (lane % 4)*2;
        if (r < M && c+1 < N) {
            C[r*N + c + 0] = acc[mi][ni][0];
            C[r*N + c + 1] = acc[mi][ni][1];
            if (r + 8 < M) {
                C[(r+8)*N + c + 0] = acc[mi][ni][2];
                C[(r+8)*N + c + 1] = acc[mi][ni][3];
            }
        }
      }
    }
}

extern "C" void launch_scaled(
    const half* A, const half* B, float* C, int M, int N, int K
) {
    dim3 block(256);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    gemm_ptx_scaled<<<grid, block>>>(A, B, C, M, N, K);
}

// ------------------------------------------------------------------
// ABI wrapper expected by test_scaled.py (ctypes looks for this name)
// ------------------------------------------------------------------
extern "C" void launch_gemm_ptx_scaled(
    const half* A, const half* B, float* C, int M, int N, int K
) {
    launch_scaled(A, B, C, M, N, K);
}
