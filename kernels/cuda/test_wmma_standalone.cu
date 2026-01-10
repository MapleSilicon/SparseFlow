#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void launch_wmma_dense(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    cudaStream_t stream
);

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void benchmark(int M, int N, int K) {
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(half);
    
    // Allocate device memory
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    // Initialize with random data
    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    for (size_t i = 0; i < M * K; i++) h_A[i] = __float2half(rand() / (float)RAND_MAX);
    for (size_t i = 0; i < K * N; i++) h_B[i] = __float2half(rand() / (float)RAND_MAX);
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_wmma_dense(d_A, d_B, d_C, M, N, K, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int iters = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        launch_wmma_dense(d_A, d_B, d_C, M, N, K, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    double avg_ms = ms / iters;
    double flops = 2.0 * M * N * K;
    double tflops = (flops / avg_ms) / 1e9;
    
    printf("Matrix: %dx%dx%d\n", M, N, K);
    printf("Time: %.3f ms\n", avg_ms);
    printf("Performance: %.2f TFLOPS\n\n", tflops);
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    printf("=== Dense WMMA Saturation Benchmark ===\n\n");
    
    // Test increasing sizes
    benchmark(512, 512, 512);
    benchmark(1024, 1024, 1024);
    benchmark(2048, 2048, 2048);
    benchmark(4096, 4096, 4096);
    
    return 0;
}
