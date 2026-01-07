#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define BM 16
#define BN 8
#define BK 32  // NOTE: Sparse K is 2× dense K!

// Minimal sparse kernel: 1 warp, 1 sparse MMA tile
// Computes C[16×8] = A_sparse[16×32] × B[32×8]
__global__ void gemm_sparse_minimal(
    const half* __restrict__ A_values,      // Compressed values (16×16 instead of 16×32)
    const uint32_t* __restrict__ A_meta,    // Metadata (2 bits per 4 values)
    const half* __restrict__ B,             // Dense B matrix
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(16) half smem_A[BM][BK/2];  // Compressed: half the K
    __shared__ __align__(16) half smem_B[BK][BN];
    __shared__ __align__(16) uint32_t smem_meta[BM][BK/16]; // 2 bits per element
    
    int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;
    
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    
    // For now, only handle K=32 (single tile)
    // Load compressed A values
    for (int i = lane; i < BM * (BK/2); i += 32) {
        int r = i / (BK/2);
        int c = i % (BK/2);
        int gr = block_row + r;
        smem_A[r][c] = (gr < M && c < K/2) ? A_values[gr * (K/2) + c] : __float2half(0.f);
    }
    
    // Load metadata
    for (int i = lane; i < BM * (BK/16); i += 32) {
        int r = i / (BK/16);
        int c = i % (BK/16);
        int gr = block_row + r;
        smem_meta[r][c] = (gr < M) ? A_meta[gr * (BK/16) + c] : 0;
    }
    
    // Load B (dense, full K=32)
    for (int i = lane; i < BK * BN; i += 32) {
        int k = i / BN;
        int n = i % BN;
        smem_B[k][n] = (k < K && block_col + n < N) ? B[k * N + block_col + n] : __float2half(0.f);
    }
    
    __syncthreads();
    
    // Load fragments
    uint32_t a[4];  // Compressed A fragment
    uint32_t e[2];  // Metadata fragment
    uint32_t b[2];  // B fragment
    
    uint32_t ptr_a = __cvta_generic_to_shared(&smem_A[lane & 15][0]);
    uint32_t ptr_e = __cvta_generic_to_shared(&smem_meta[lane & 15][0]);
    uint32_t ptr_b = __cvta_generic_to_shared(&smem_B[lane & 15][0]);
    
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(ptr_a)
    );
    
    // Load metadata (2×16 bits = 32 bits per register)
    asm volatile(
        "ldmatrix.sync.aligned.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(e[0]), "=r"(e[1])
        : "r"(ptr_e)
    );
    
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(ptr_b)
    );
    
    // SPARSE MMA INSTRUCTION!
    // Key difference: .sp modifier and k32 instead of k16
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10,%11}, 0x0;\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(e[0]), "r"(e[1])  // Metadata registers
    );
    
    // Store
    int row = block_row + (lane >> 2);
    int col = block_col + (lane & 3) * 2;
    
    if (row < M && col + 1 < N) {
        C[row * N + col + 0] = acc[0];
        C[row * N + col + 1] = acc[1];
    }
    if (row + 8 < M && col + 1 < N) {
        C[(row + 8) * N + col + 0] = acc[2];
        C[(row + 8) * N + col + 1] = acc[3];
    }
}

extern "C" void launch_gemm_sparse_minimal(
    const half* A_values, const uint32_t* A_meta, const half* B, float* C,
    int M, int N, int K
) {
    dim3 block(32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_sparse_minimal<<<grid, block>>>(A_values, A_meta, B, C, M, N, K);
}
