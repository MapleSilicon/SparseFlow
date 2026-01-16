#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ void decode_2of4(uint16_t code, int &i0, int &i1) {
  switch (code) {
    case 0: i0 = 0; i1 = 1; break;
    case 1: i0 = 0; i1 = 2; break;
    case 2: i0 = 0; i1 = 3; break;
    case 3: i0 = 1; i1 = 2; break;
    case 4: i0 = 1; i1 = 3; break;
    case 5: i0 = 2; i1 = 3; break;
    default: i0 = 0; i1 = 2; break;
  }
}

// Acomp: [M, K/2] row-major
// E:     stored as [K/4, M] (K-major) => index as E[g*M + row]
// B:     [K, N] row-major (confirmed by your CPU check)
__global__ void sparse_gemm_A_sparse_B_dense_ref_v3(
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
  int G = K / 4;

  for (int g = 0; g < G; g++) {
    // Acomp has 2 values per group
    half a0h = Acomp[row * (K/2) + g*2 + 0];
    half a1h = Acomp[row * (K/2) + g*2 + 1];
    float a0 = __half2float(a0h);
    float a1 = __half2float(a1h);

    // *** CRITICAL FIX: E is K-major ***
    uint16_t code = E[g * M + row];

    int i0, i1;
    decode_2of4(code, i0, i1);

    int k0 = g*4 + i0;
    int k1 = g*4 + i1;

    // B is row-major: B[k, col] == B[k*N + col]
    float b0 = __half2float(B[k0 * N + col]);
    float b1 = __half2float(B[k1 * N + col]);

    acc += a0 * b0 + a1 * b1;
  }

  C[row * N + col] = acc;
}

extern "C" void launch_sparse_ref_v3(
    const half* Acomp, const uint16_t* E, const half* B, float* C,
    int M, int N, int K)
{
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y);
  sparse_gemm_A_sparse_B_dense_ref_v3<<<grid, block>>>(Acomp, E, B, C, M, N, K);
}
