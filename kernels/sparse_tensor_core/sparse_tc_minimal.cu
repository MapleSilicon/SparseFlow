#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void sparse_tc_minimal(
    const half* __restrict__ A,
    const uint16_t* __restrict__ E,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Each thread computes ONE output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Loop over K in groups of 4
    for (int k = 0; k < K; k += 4) {
        int g = k / 4;
        
        // Get metadata for this row and group
        uint16_t code = E[g * M + row];
        
        // Decode which 2 of 4 are kept
        int i0, i1;
        switch(code) {
            case 0: i0=0; i1=1; break;
            case 1: i0=0; i1=2; break;
            case 2: i0=0; i1=3; break;
            case 3: i0=1; i1=2; break;
            case 4: i0=1; i1=3; break;
            case 5: i0=2; i1=3; break;
            default: i0=0; i1=2; break;
        }
        
        // Load compressed A values
        half a0 = A[row * (K/2) + g*2 + 0];
        half a1 = A[row * (K/2) + g*2 + 1];
        
        // Load B values at the kept positions
        half b0 = B[(k + i0) * N + col];
        half b1 = B[(k + i1) * N + col];
        
        // Accumulate
        sum += __half2float(a0) * __half2float(b0);
        sum += __half2float(a1) * __half2float(b1);
    }
    
    C[row * N + col] = sum;
}

extern "C" void launch_sparse_tc_minimal(
    const half* A, const uint16_t* E, const half* B, float* C,
    int M, int N, int K)
{
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    sparse_tc_minimal<<<grid, block>>>(A, E, B, C, M, N, K);
}
