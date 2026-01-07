#include <cusparseLt.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../benchmark/micro_bench.h"

namespace sparseflow {

static cusparseLtHandle_t* g_cusparselt_handle = nullptr;

static bool ensure_cusparselt_handle() {
  if (g_cusparselt_handle == nullptr) {
    g_cusparselt_handle = new cusparseLtHandle_t;
    cusparseStatus_t status = cusparseLtInit(g_cusparselt_handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      std::cerr << "[cuSPARSELt] Init failed (status=" << status << ")" << std::endl;
      delete g_cusparselt_handle;
      g_cusparselt_handle = nullptr;
      return false;
    }
  }
  return true;
}

// Real 2:4 sparse matmul using cuSPARSELt
bool launch_sparse_nm_32(const MatmulDesc& desc,
                         void* A, void* B, void* C,
                         cudaStream_t stream) {
  
  if (!ensure_cusparselt_handle()) {
    return false;
  }
  
  cusparseLtMatDescriptor_t matA, matB, matC, matD;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  
  cusparseOrder_t order = CUSPARSE_ORDER_ROW;
  cusparseComputeType compute_type = CUSPARSE_COMPUTE_32F;
  
  cudaDataType_t cuda_dtype;
  switch (desc.dtype) {
    case DType::F16:  cuda_dtype = CUDA_R_16F; break;
    case DType::BF16: cuda_dtype = CUDA_R_16BF; break;
    case DType::F32:  cuda_dtype = CUDA_R_32F; break;
    default:          cuda_dtype = CUDA_R_16F; break;
  }
  
  unsigned alignment = 16;
  
  cusparseStatus_t status;
  
  // Initialize sparse matrix A (M x K, 2:4 structured)
  status = cusparseLtStructuredDescriptorInit(
    g_cusparselt_handle, &matA,
    desc.M, desc.K, desc.K,
    alignment, cuda_dtype, order,
    CUSPARSELT_SPARSITY_50_PERCENT
  );
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "[cuSPARSELt] Failed to init matA (status=" << status << ")" << std::endl;
    return false;
  }
  
  // Initialize dense matrix B (K x N)
  status = cusparseLtDenseDescriptorInit(
    g_cusparselt_handle, &matB,
    desc.K, desc.N, desc.N,
    alignment, cuda_dtype, order
  );
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "[cuSPARSELt] Failed to init matB" << std::endl;
    cusparseLtMatDescriptorDestroy(&matA);
    return false;
  }
  
  // Initialize dense matrix C (M x N, output)
  status = cusparseLtDenseDescriptorInit(
    g_cusparselt_handle, &matC,
    desc.M, desc.N, desc.N,
    alignment, cuda_dtype, order
  );
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "[cuSPARSELt] Failed to init matC" << std::endl;
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    return false;
  }
  
  // Initialize matD (same as matC)
  status = cusparseLtDenseDescriptorInit(
    g_cusparselt_handle, &matD,
    desc.M, desc.N, desc.N,
    alignment, cuda_dtype, order
  );
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "[cuSPARSELt] Failed to init matD" << std::endl;
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    return false;
  }
  
  // Create matmul descriptor
  status = cusparseLtMatmulDescriptorInit(
    g_cusparselt_handle, &matmul,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &matA, &matB, &matC, &matD,
    compute_type
  );
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "[cuSPARSELt] Failed to init matmul descriptor" << std::endl;
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    return false;
  }
  
  // Algorithm selection
  status = cusparseLtMatmulAlgSelectionInit(
    g_cusparselt_handle, &alg_sel, &matmul,
    CUSPARSELT_MATMUL_ALG_DEFAULT
  );
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "[cuSPARSELt] Failed to init alg selection" << std::endl;
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    return false;
  }
  
  // Create plan
  status = cusparseLtMatmulPlanInit(
    g_cusparselt_handle, &plan, &matmul, &alg_sel
  );
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "[cuSPARSELt] Failed to init plan" << std::endl;
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    return false;
  }
  
  // Get workspace size
  size_t workspace_size = 0;
  status = cusparseLtMatmulGetWorkspace(g_cusparselt_handle, &plan, &workspace_size);
  
  void* d_workspace = nullptr;
  if (workspace_size > 0) {
    cudaMalloc(&d_workspace, workspace_size);
  }
  
  float alpha = 1.0f;
  float beta = 0.0f;
  
  // Execute sparse matmul
  // Note: A should be in compressed 2:4 format
  // For Phase-1, we're passing dense A which will cause an error
  // Real implementation needs pruning + compression step
  status = cusparseLtMatmul(
    g_cusparselt_handle, &plan,
    &alpha, A, B, &beta, C, C,
    d_workspace, &stream, 1
  );
  
  // Cleanup
  if (d_workspace) cudaFree(d_workspace);
  cusparseLtMatmulPlanDestroy(&plan);
  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matB);
  cusparseLtMatDescriptorDestroy(&matC);
  cusparseLtMatDescriptorDestroy(&matD);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "[cuSPARSELt] Matmul failed (status=" << status << ")" << std::endl;
    std::cerr << "[cuSPARSELt] Note: Input A must be in compressed 2:4 format" << std::endl;
    return false;
  }
  
  return true;
}

} // namespace sparseflow
