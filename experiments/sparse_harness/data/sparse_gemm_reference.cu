#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ inline void decode_2of4(uint8_t code, int &i0, int &i1) {
  switch (code & 0x3) {
    case 0: i0 = 0; i1 = 1; break;
    case 1: i0 = 0; i1 = 2; break;
    case 2: i0 = 1; i1 = 3; break;
    case 3: i0 = 2; i1 = 3; break;
  }
}

// A is sparse compressed (2 values per 4), meta packed 4-bit per group, B is dense ColumnMajor.
__global__ void sparse_gemm_A_sparse_B_dense(
    const half* __restrict__ Acomp,   // [M, K/2] row-major
    const uint8_t* __restrict__ E,    // [M, K/8] bytes, each byte = 2 nibbles (2 groups)
    const half* __restrict__ B,       // [K, N] column-major (B[k + col*K])
    float* __restrict__ C,            // [M, N] row-major
    int M, int N, int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;

  int groups = K / 4;       // e.g. 32
  int bytes_per_row = K / 8; // e.g. 16 (2 groups per byte)

  for (int g = 0; g < groups; g++) {
    // metadata nibble for this 4-wide group
    uint8_t byte = E[row * bytes_per_row + (g >> 1)];
    uint8_t code = (g & 1) ? (byte >> 4) : (byte & 0xF);

    int i0, i1;
    decode_2of4(code, i0, i1);

    // read the 2 stored values for this group
    int a_base = row * (K/2) + g * 2;
    half av0 = Acomp[a_base + 0];
    half av1 = Acomp[a_base + 1];

    int k0 = g * 4 + i0;
    int k1 = g * 4 + i1;

    // B is column-major
    half bv0 = B[k0 + col * K];
    half bv1 = B[k1 + col * K];

    acc += __half2float(av0) * __half2float(bv0);
    acc += __half2float(av1) * __half2float(bv1);
  }

  C[row * N + col] = acc;
}

extern "C" void launch_sparse_gemm_reference(
    const half* Acomp,
    const uint8_t* E,
    const half* B,
    float* C,
    int M, int N, int K)
{
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  sparse_gemm_A_sparse_B_dense<<<grid, block>>>(Acomp, E, B, C, M, N, K);
}
