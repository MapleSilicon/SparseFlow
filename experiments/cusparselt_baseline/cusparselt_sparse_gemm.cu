#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(x) do { if((x)!=cudaSuccess){printf("CUDA error\n"); exit(1);} } while(0)
#define CHECK_LT(x) do { if((x)!=CUSPARSE_STATUS_SUCCESS){printf("cuSPARSELt error\n"); exit(1);} } while(0)

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    const size_t sizeA = M * K * sizeof(__half);
    const size_t sizeB = K * N * sizeof(__half);
    const size_t sizeC = M * N * sizeof(float);

    __half *A, *B;
    float *C;

    CHECK_CUDA(cudaMalloc(&A, sizeA));
    CHECK_CUDA(cudaMalloc(&B, sizeB));
    CHECK_CUDA(cudaMalloc(&C, sizeC));

    // Init random
    CHECK_CUDA(cudaMemset(A, 1, sizeA));
    CHECK_CUDA(cudaMemset(B, 1, sizeB));
    CHECK_CUDA(cudaMemset(C, 0, sizeC));

    cusparseLtHandle_t handle;
    CHECK_LT(cusparseLtInit(&handle));

    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg;
    cusparseLtMatmulPlan_t plan;

    // A: FP16, sparse, row-major
    CHECK_LT(cusparseLtStructuredDescriptorInit(
        &handle, &matA,
        M, K, K,
        CUSPARSE_ORDER_ROW,
        CUDA_R_16F,
        CUSPARSELT_SPARSITY_50_PERCENT));

    // B: FP16, dense, column-major
    CHECK_LT(cusparseLtDenseDescriptorInit(
        &handle, &matB,
        K, N, K,
        CUSPARSE_ORDER_COL,
        CUDA_R_16F));

    // C: FP32
    CHECK_LT(cusparseLtDenseDescriptorInit(
        &handle, &matC,
        M, N, N,
        CUSPARSE_ORDER_ROW,
        CUDA_R_32F));

    CHECK_LT(cusparseLtMatmulDescriptorInit(
        &handle, &matmul,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matA, &matB, &matC, &matC,
        CUSPARSE_COMPUTE_32F));

    CHECK_LT(cusparseLtMatmulAlgSelectionInit(
        &handle, &alg, &matmul,
        CUSPARSELT_MATMUL_ALG_DEFAULT));

    size_t workspaceSize;
    CHECK_LT(cusparseLtMatmulGetWorkspace(
        &handle, &alg, &workspaceSize));

    void* workspace;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    CHECK_LT(cusparseLtMatmulPlanInit(
        &handle, &plan, &matmul, &alg, workspaceSize));

    // 🔥 PRUNE + PACK A (THIS IS THE MAGIC)
    CHECK_LT(cusparseLtSpMMAPrune(
        &handle, &matA, A, A, CUSPARSELT_PRUNE_SPMMA_TILE, 0));

    CHECK_LT(cusparseLtSpMMACompress(
        &handle, &matA, A, A, 0));

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for(int i=0;i<5;i++)
        CHECK_LT(cusparseLtMatmul(
            &handle, &plan,
            &alpha, A, B,
            &beta, C, C,
            workspace, nullptr));

    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i=0;i<20;i++)
        CHECK_LT(cusparseLtMatmul(
            &handle, &plan,
            &alpha, A, B,
            &beta, C, C,
            workspace, nullptr));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 20.0f;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms * 1e9);

    printf("Sparse GEMM latency: %.3f ms\n", ms);
    printf("Sparse GEMM TFLOPS:  %.2f\n", tflops);

    return 0;
}
