#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 16

extern "C" __global__ void gemm_ptx_lone_warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ alignas(16) half smem_A[BM][BK];
    __shared__ alignas(16) half smem_B[BK][BN];

    const int lane = threadIdx.x & 31;
    
    // 8×16 MMA grid (entire 128×128)
    float acc[8][16][4];
    for(int i=0; i<8; ++i) 
        for(int j=0; j<16; ++j) 
            for(int k=0; k<4; ++k) 
                acc[i][j][k] = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        
        // Single warp loads entire tile
        for (int i = lane; i < (BM * BK); i += 32) {
            int r = i / BK;
            int c = i % BK;
            smem_A[r][c] = (r < M && (k_tile + c) < K) ? 
                A[r * K + (k_tile + c)] : __float2half(0.0f);
        }
        
        for (int i = lane; i < (BK * BN); i += 32) {
            int r = i / BN;
            int c = i % BN;
            smem_B[r][c] = ((k_tile + r) < K && c < N) ? 
                B[(k_tile + r) * N + c] : __float2half(0.0f);
        }
        
        __syncthreads();

        // 8×16 MMA grid
        #pragma unroll
        for (int mi = 0; mi < 8; mi++) {
            uint32_t a[4];
            uint32_t ptr_a = __cvta_generic_to_shared(
                &smem_A[mi * 16 + (lane % 16)][(lane / 16) * 8]
            );
            
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                : "r"(ptr_a)
            );

            #pragma unroll
            for (int ni = 0; ni < 16; ni++) {
                uint32_t b[2];
                uint32_t ptr_b = __cvta_generic_to_shared(
                    &smem_B[lane % 16][ni * 8]
                );
                
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                    : "=r"(b[0]), "=r"(b[1])
                    : "r"(ptr_b)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                      "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(b[0]), "r"(b[1])
                );
            }
        }
        
        __syncthreads();
    }

    // Store 128×128
    for (int mi = 0; mi < 8; mi++) {
        for (int ni = 0; ni < 16; ni++) {
            int r = (mi * 16) + (lane / 4);
            int c = (ni * 8) + (lane % 4) * 2;
            
            if (r < M && c + 1 < N) {
                C[r * N + c + 0] = acc[mi][ni][0];
                C[r * N + c + 1] = acc[mi][ni][1];
            }
            if (r + 8 < M && c + 1 < N) {
                C[(r + 8) * N + c + 0] = acc[mi][ni][2];
                C[(r + 8) * N + c + 1] = acc[mi][ni][3];
            }
        }
    }
}

extern "C" void launch_gemm_ptx_lone_warp(
    const half* A, const half* B, float* C, int M, int N, int K
) {
    dim3 block(32);
    dim3 grid(1, 1);  // Single block, single warp
    gemm_ptx_lone_warp<<<grid, block>>>(A, B, C, M, N, K);
}
