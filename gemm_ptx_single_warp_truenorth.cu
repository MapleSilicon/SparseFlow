#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void gemm_ptx_single_warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ alignas(16) half smem_A[32][16];
    __shared__ alignas(16) half smem_B[16][32];

    const int lane = threadIdx.x & 31;
    
    int block_row = blockIdx.y * 32;
    int block_col = blockIdx.x * 32;
    
    float acc[2][4][4] = {0.0f};

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        
        // Simple load: each thread loads its row
        for(int i = 0; i < 16; i++) {
            if (lane < 32 && block_row + lane < M && k_tile + i < K) {
                smem_A[lane][i] = A[(block_row + lane) * K + (k_tile + i)];
            }
            if (lane < 32 && k_tile + i < K && block_col + lane < N) {
                smem_B[i][lane] = B[(k_tile + i) * N + (block_col + lane)];
            }
        }
        __syncthreads();

        // 2×4 MMA grid
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
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
            for (int ni = 0; ni < 4; ni++) {
                uint32_t b[2];
                
                // KEY: Row pointers for .trans
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

    // Store
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        for (int ni = 0; ni < 4; ni++) {
            
            int r = block_row + (mi * 16) + (lane / 4);
            int c = block_col + (ni * 8) + (lane % 4) * 2;
            
            if (r < M && c + 1 < N) {
                C[r * N + c + 0] = acc[mi][ni][0];
                C[r * N + c + 1] = acc[mi][ni][1];
                
                if (r + 8 < M) {
                    C[(r + 8) * N + c + 0] = acc[mi][ni][2];
                    C[(r + 8) * N + c + 1] = acc[mi][ni][3];
                }
            }
        }
    }
}

extern "C" void launch_gemm_ptx_single_warp(
    const half* A, const half* B, float* C, int M, int N, int K
) {
    dim3 block(32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    gemm_ptx_single_warp<<<grid, block>>>(A, B, C, M, N, K);
}
