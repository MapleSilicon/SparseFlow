#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <stdio.h>

extern "C" int benchmark_cusparselt_24(
    const half* A, const half* B, half* C,
    int M, int N, int K
) {
    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    
    // Initialize
    cusparseLtInit(&handle);
    
    // Create matrix descriptors
    cusparseLtStructuredDescriptorInit(
        &handle, &matA, M, K, K, 16,
        CUDA_R_16F, CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT
    );
    
    cusparseLtDenseDescriptorInit(
        &handle, &matB, K, N, N, 16,
        CUDA_R_16F, CUSPARSE_ORDER_ROW
    );
    
    cusparseLtDenseDescriptorInit(
        &handle, &matC, M, N, N, 16,
        CUDA_R_16F, CUSPARSE_ORDER_ROW
    );
    
    // Create matmul descriptor
    cusparseLtMatmulDescriptorInit(
        &handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &matA, &matB, &matC, &matC,
        CUSPARSE_COMPUTE_16F
    );
    
    // Algorithm selection
    cusparseLtMatmulAlgSelectionInit(
        &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT
    );
    
    // Create plan
    size_t workspace_size;
    cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
    
    void* workspace;
    cudaMalloc(&workspace, workspace_size);
    
    // Execute
    float alpha = 1.0f, beta = 0.0f;
    cusparseLtMatmul(
        &handle, &plan, &alpha, A, B, &beta, C, C,
        workspace, nullptr, 0
    );
    
    // Cleanup
    cudaFree(workspace);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtDestroy(&handle);
    
    return 0;
}
