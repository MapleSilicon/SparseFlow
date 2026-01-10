#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <cstdio>

#define CHK(x) do { auto err = (x); if(err != CUSPARSE_STATUS_SUCCESS) { printf("Error at %d\n", __LINE__); exit(1); } } while(0)

int main() {
    int M = 4096, N = 4096, K = 4096;
    
    __half *A, *B, *Ac;
    float *C;
    
    cudaMalloc(&A, M*K*sizeof(__half));
    cudaMalloc(&B, K*N*sizeof(__half));
    cudaMalloc(&Ac, M*K/2*sizeof(__half));
    cudaMalloc(&C, M*N*sizeof(float));
    
    cudaMemset(A, 1, M*K*sizeof(__half));
    cudaMemset(B, 1, K*N*sizeof(__half));
    
    cusparseLtHandle_t h;
    cusparseLtInit(&h);
    
    cusparseLtMatDescriptor_t mA, mB, mC;
    
    CHK(cusparseLtStructuredDescriptorInit(&h, &mA, M, K, K, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));
    CHK(cusparseLtDenseDescriptorInit(&h, &mB, K, N, N, 16, CUDA_R_16F, CUSPARSE_ORDER_COL));
    CHK(cusparseLtDenseDescriptorInit(&h, &mC, M, N, N, 16, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    
    cusparseLtMatmulDescriptor_t mm;
    CHK(cusparseLtMatmulDescriptorInit(&h, &mm, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &mA, &mB, &mC, &mC, CUSPARSE_COMPUTE_32F));
    
    cusparseLtMatmulAlgSelection_t alg;
    CHK(cusparseLtMatmulAlgSelectionInit(&h, &alg, &mm, CUSPARSELT_MATMUL_ALG_DEFAULT));
    
    cusparseLtMatmulPlan_t plan;
    CHK(cusparseLtMatmulPlanInit(&h, &plan, &mm, &alg));
    
    size_t ws_size;
    CHK(cusparseLtMatmulGetWorkspace(&h, &plan, &ws_size));
    void *ws;
    cudaMalloc(&ws, ws_size);
    
    CHK(cusparseLtSpMMAPrune(&h, &mm, A, Ac, CUSPARSELT_PRUNE_SPMMA_TILE, 0));
    CHK(cusparseLtSpMMACompress(&h, &plan, Ac, Ac, 0));
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for(int i=0; i<5; i++)
        CHK(cusparseLtMatmul(&h, &plan, &alpha, Ac, B, &beta, C, C, ws, nullptr, 0));
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i=0; i<20; i++)
        CHK(cusparseLtMatmul(&h, &plan, &alpha, Ac, B, &beta, C, C, ws, nullptr, 0));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 20.0f;
    
    double tflops = (2.0 * M * N * K) / (ms * 1e9);
    
    printf("\n========================================\n");
    printf("cuSPARSELt 2:4 Sparse Benchmark\n");
    printf("========================================\n");
    printf("Size: %dx%dx%d\n", M, N, K);
    printf("Time: %.3f ms\n", ms);
    printf("TFLOPS: %.2f\n", tflops);
    printf("========================================\n");
    
    if(tflops > 150) printf("✓ Real sparse acceleration!\n");
    else if(tflops > 100) printf("⚠ Might be running dense\n");
    else printf("✗ Performance issue\n");
    
    return 0;
}
