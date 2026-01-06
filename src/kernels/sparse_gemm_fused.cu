// src/kernels/sparse_gemm_fused.cu
// Fused sparse GEMM with configurable epilogue

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Simple version for now - we'll expand later
namespace sparseflow {

enum class EpilogueKind : int {
    NONE = 0,
    RELU = 1,
    SILU = 2,
};

// Basic kernel without epilogue (MVP)
__global__ void sparse_gemm_basic(
    const half* __restrict__ A,
    const half* __restrict__ Bc,
    half* __restrict__ C,
    int M, int N, int K
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int m = tid / N;
    int n = tid % N;
    
    if (m < M && n < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[m * K + k]) * __half2float(Bc[k * N + n]);
        }
        C[m * N + n] = __float2half(sum);
    }
}

// Launcher
extern "C" 
cudaError_t launch_sparse_gemm(
    const half* A, const half* Bc, half* C,
    int M, int N, int K, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    
    sparse_gemm_basic<<<blocks, threads, 0, stream>>>(A, Bc, C, M, N, K);
    
    return cudaGetLastError();
}

} // namespace sparseflow
