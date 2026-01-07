#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 16
#define WARPS 8
#define THREADS (WARPS * 32)

__device__ __forceinline__ void cp_async_ca_16B(uint32_t smem_addr, const void* gmem_addr) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_addr)
    );
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_0() { asm volatile("cp.async.wait_group 0;\n"); }

extern "C" __global__
void gemm_ptx_scaled_cpasync(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Double-buffered shared memory
    __shared__ alignas(16) half smem_A[2][BM][BK];   // [2][128][16]
    __shared__ alignas(16) half smem_B[2][BK][BN];   // [2][16][128]

    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid >> 5;

    // Warp tile geometry (same as your correct non-cpasync kernel)
    int warp_m = warp_id >> 1;   // 0..3
    int warp_n = warp_id & 1;    // 0..1
    int warp_row = warp_m * 32;  // 0,32,64,96
    int warp_col = warp_n * 64;  // 0 or 64

    int block_row = (int)blockIdx.y * BM;
    int block_col = (int)blockIdx.x * BN;

    float acc[2][8][4];
    #pragma unroll
    for (int mi=0; mi<2; mi++)
        for (int ni=0; ni<8; ni++)
            for (int t=0; t<4; t++)
                acc[mi][ni][t] = 0.0f;

    auto prefetch_stage = [&](int stage, int k0) {
        // We ONLY issue cp.async on 16B-aligned chunks:
        // A: each row has 16 half = 32B => 2 chunks per row (cols 0 and 8)
        // Total A chunks = 128 rows * 2 = 256 chunks
        int a_chunk = tid; // 0..255 exactly
        {
            int r = a_chunk >> 1;          // 0..127
            int c8 = (a_chunk & 1) * 8;    // 0 or 8
            int gr = block_row + r;
            int gc = k0 + c8;

            uint32_t smem_addr = __cvta_generic_to_shared(&smem_A[stage][r][c8]);

            if (gr < M && (gc + 7) < K) {
                // 16B from global: 8 halfs
                cp_async_ca_16B(smem_addr, (const void*)(&A[gr * K + gc]));
            }
        }

        // B: each row has 128 half = 256B => 16 chunks per row (cols 0,8,16,...,120)
        // Total B chunks = 16 rows * 16 = 256 chunks
        int b_chunk = tid; // 0..255 exactly
        {
            int r  = b_chunk >> 4;         // 0..15
            int c8 = (b_chunk & 15) * 8;   // 0..120 step 8
            int gr = k0 + r;
            int gc = block_col + c8;

            uint32_t smem_addr = __cvta_generic_to_shared(&smem_B[stage][r][c8]);

            if (gr < K && (gc + 7) < N) {
                cp_async_ca_16B(smem_addr, (const void*)(&B[gr * N + gc]));
            }
        }

        cp_async_commit();
    };

    int stage = 0;

    // Prefetch first tile
    prefetch_stage(stage, 0);
    cp_async_wait_0();
    __syncthreads();

    // Zero-fill any out-of-bounds 16B chunks we didn't fetch
    // (fast enough for now; later we can use zfill variants)
    {
        int a_chunk = tid;
        int r = a_chunk >> 1;
        int c8 = (a_chunk & 1) * 8;
        int gr = block_row + r;
        int gc = c8;
        if (!(gr < M && (gc + 7) < K)) {
            #pragma unroll
            for (int i=0;i<8;i++) smem_A[stage][r][c8 + i] = __float2half(0.0f);
        }

        int b_chunk = tid;
        int rr  = b_chunk >> 4;
        int cc8 = (b_chunk & 15) * 8;
        int grb = rr;
        int gcb = block_col + cc8;
        if (!(grb < K && (gcb + 7) < N)) {
            #pragma unroll
            for (int i=0;i<8;i++) smem_B[stage][rr][cc8 + i] = __float2half(0.0f);
        }
    }
    __syncthreads();

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        int next_k = k_tile + BK;
        int next = stage ^ 1;

        // Prefetch next tile while computing current
        if (next_k < K) {
            prefetch_stage(next, next_k);
        }

        // --- Compute on current stage ---
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            uint32_t a[4];
            uint32_t ptr_a = __cvta_generic_to_shared(
                &smem_A[stage][warp_row + mi*16 + (lane % 16)][(lane / 16) * 8]
            );
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                : "r"(ptr_a)
            );

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                uint32_t bregs[2];
                uint32_t ptr_b = __cvta_generic_to_shared(
                    &smem_B[stage][lane % 16][warp_col + ni*8]
                );

                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(bregs[0]), "=r"(bregs[1])
                    : "r"(ptr_b)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, "
                    "{%4,%5,%6,%7}, "
                    "{%8,%9}, "
                    "{%0,%1,%2,%3};\n"
                    : "+f"(acc[mi][ni][0]),
                      "+f"(acc[mi][ni][1]),
                      "+f"(acc[mi][ni][2]),
                      "+f"(acc[mi][ni][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(bregs[0]), "r"(bregs[1])
                );
            }
        }

        // Stage swap
        if (next_k < K) {
            cp_async_wait_0();
            __syncthreads();

            // Zero-fill OOB for next stage
            int a_chunk = tid;
            int r = a_chunk >> 1;
            int c8 = (a_chunk & 1) * 8;
            int gr = block_row + r;
            int gc = next_k + c8;
            if (!(gr < M && (gc + 7) < K)) {
                #pragma unroll
                for (int i=0;i<8;i++) smem_A[next][r][c8 + i] = __float2half(0.0f);
            }

            int b_chunk = tid;
            int rr  = b_chunk >> 4;
            int cc8 = (b_chunk & 15) * 8;
            int grb = next_k + rr;
            int gcb = block_col + cc8;
            if (!(grb < K && (gcb + 7) < N)) {
                #pragma unroll
                for (int i=0;i<8;i++) smem_B[next][rr][cc8 + i] = __float2half(0.0f);
            }

            __syncthreads();
            stage = next;
        }
    }

    // --- Store ---
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int r = block_row + warp_row + mi*16 + (lane / 4);
            int c = block_col + warp_col + ni*8 + (lane % 4)*2;
            if (r < M && (c + 1) < N) {
                C[r*N + c + 0] = acc[mi][ni][0];
                C[r*N + c + 1] = acc[mi][ni][1];
                if ((r + 8) < M) {
                    C[(r+8)*N + c + 0] = acc[mi][ni][2];
                    C[(r+8)*N + c + 1] = acc[mi][ni][3];
                }
            }
        }
    }
}

extern "C" void launch_gemm_ptx_scaled(
    const half* A, const half* B, float* C, int M, int N, int K
) {
    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_ptx_scaled_cpasync<<<grid, block>>>(A, B, C, M, N, K);
}
