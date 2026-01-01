#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void launch_gemm_pipeline(
    const half* A, const half* B, float* C,
    int M, int N, int K, cudaStream_t stream
);

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

int main() {
    // Edge shapes that catch boundary bugs
    const int shapes[][3] = {
        {16,16,16},
        {127,127,127},
        {129,257,65},
        {1023,2049,511},
    };

    for (auto &s : shapes) {
        int M = s[0], N = s[1], K = s[2];
        size_t bytesA = (size_t)M*K*sizeof(half);
        size_t bytesB = (size_t)K*N*sizeof(half);
        size_t bytesC = (size_t)M*N*sizeof(float);

        half *dA=nullptr, *dB=nullptr;
        float *dC=nullptr;
        checkCuda(cudaMalloc(&dA, bytesA), "malloc A");
        checkCuda(cudaMalloc(&dB, bytesB), "malloc B");
        checkCuda(cudaMalloc(&dC, bytesC), "malloc C");
        checkCuda(cudaMemset(dC, 0, bytesC), "memset C");

        cudaStream_t stream;
        checkCuda(cudaStreamCreate(&stream), "stream create");

        // Launch (kernel should be boundary-safe)
        launch_gemm_pipeline(dA, dB, dC, M, N, K, stream);
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaStreamSynchronize(stream), "sync");

        // If we got here without illegal access, boundary safety is working.
        printf("[OK] launch_gemm_pipeline for M=%d N=%d K=%d\n", M, N, K);

        cudaStreamDestroy(stream);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    printf("✅ All launches completed without CUDA errors.\n");
    return 0;
}
