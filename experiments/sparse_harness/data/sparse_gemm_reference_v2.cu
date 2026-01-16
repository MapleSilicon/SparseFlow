#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ void decode_2of4(uint16_t code, int &i0, int &i1) {
  // 6 combos of choosing 2 out of 4
  switch (code) {
    case 0: i0 = 0; i1 = 1; break;
    case 1: i0 = 0; i1 = 2; break;
    case 2: i0 = 0; i1 = 3; break;
    case 3: i0 = 1; i1 = 2; break;
    case 4: i0 = 1; i1 = 3; break;
    case 5: i0 = 2; i1 = 3; break;
    default: i0 = 0; i1 = 2; break; // safe fallback
  }
}

// Acomp: [M, K/2] row-major (2 values per 4)
// E:     [M, K/4] uint16 codes (one code per 4)
// B:     [K, N]  (choose ONE indexing below based on your step (1))
__global__ void sparse_gemm_A_sparse_B_dense_ref(
    const half* __restrict__ Acomp,
    const uint16_t* __restrict__ E,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;

  int groups = K / 4;              // 32 for K=128
  int a_row_base = row * (K/2);    // 64 halfs per row
  int e_row_base = row * groups;   // 32 uint16 codes per row

  #pragma unroll
  for (int g = 0; g < groups; g++) {
    uint16_t code = E[e_row_base + g];

    int i0, i1;
    decode_2of4(code, i0, i1);

    // two stored values for this group (packed in order matching the codebook)
    half a0 = Acomp[a_row_base + g*2 + 0];
    half a1 = Acomp[a_row_base + g*2 + 1];

    int k0 = g*4 + i0;
    int k1 = g*4 + i1;

    // === PICK THE CORRECT B INDEXING ===
    // Row-major B:
    half b0 = B[k0 * N + col];
    half b1 = B[k1 * N + col];

    // If your step (1) says COLUMN-MAJOR, replace the 2 lines above with:
    // half b0 = B[k0 + col * K];
    // half b1 = B[k1 + col * K];

    acc += __half2float(a0) * __half2float(b0);
    acc += __half2float(a1) * __half2float(b1);
  }

  C[row * N + col] = acc;
}

extern "C" void launch_sparse_ref_v2(
    const half* Acomp, const uint16_t* E, const half* B, float* C,
    int M, int N, int K)
{
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
  sparse_gemm_A_sparse_B_dense_ref<<<grid, block>>>(Acomp, E, B, C, M, N, K);
}
