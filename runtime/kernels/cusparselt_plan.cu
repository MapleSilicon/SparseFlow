#include "cusparselt_plan.h"
#include "../runtime_state/cusparselt_context.h"
#include <iostream>

#define CHECK_CUDA(expr) do {                                   \
  cudaError_t _e = (expr);                                      \
  if (_e != cudaSuccess) {                                      \
    std::cerr << "[cuSPARSELt Plan] CUDA error: "               \
              << cudaGetErrorString(_e) << "\n";                \
    return false;                                               \
  }                                                            \
} while (0)

#define CHECK_LT(expr) do {                                     \
  cusparseStatus_t _s = (expr);                                 \
  if (_s != CUSPARSE_STATUS_SUCCESS) {                          \
    std::cerr << "[cuSPARSELt Plan] cuSPARSE error: "           \
              << (int)_s << "\n";                               \
    return false;                                               \
  }                                                            \
} while (0)

namespace sparseflow {

CusparseLtPlan::~CusparseLtPlan() {
  if (d_valid) { cudaFree(d_valid); d_valid = nullptr; }
  if (d_compressed) { cudaFree(d_compressed); d_compressed = nullptr; }
  if (workspace) { cudaFree(workspace); workspace = nullptr; }

  if (valid) {
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    valid = false;
  }
}

bool CusparseLtPlan::init(int M_, int N_, int K_) {
  M = M_; N = N_; K = K_;

  auto& ctx = get_cusparselt_context();
  if (!ctx.initialized) return false;
  
  handle = ctx.handle;

  // Use EXACT signatures from working code
  CHECK_LT(cusparseLtStructuredDescriptorInit(
      &handle, &matA, M, K, K, 16,
      CUDA_R_16F, CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));

  CHECK_LT(cusparseLtDenseDescriptorInit(
      &handle, &matB, K, N, N, 16,
      CUDA_R_16F, CUSPARSE_ORDER_ROW));

  CHECK_LT(cusparseLtDenseDescriptorInit(
      &handle, &matC, M, N, N, 16,
      CUDA_R_16F, CUSPARSE_ORDER_ROW));

  CHECK_LT(cusparseLtMatmulDescriptorInit(
      &handle, &matmulDesc,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &matA, &matB, &matC, &matC,
      CUSPARSE_COMPUTE_32F));

  CHECK_LT(cusparseLtMatmulAlgSelectionInit(
      &handle, &algSel, &matmulDesc,
      CUSPARSELT_MATMUL_ALG_DEFAULT));

  CHECK_LT(cusparseLtMatmulPlanInit(&handle, &plan, &matmulDesc, &algSel));

  CHECK_LT(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));
  if (workspace_size > 0) {
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
  }

  CHECK_CUDA(cudaMalloc((void**)&d_valid, sizeof(int)));

  size_t compressed_buffer_size = 0;
  CHECK_LT(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compressed_buffer_size));
  
  CHECK_CUDA(cudaMalloc(&d_compressed, compressed_size + compressed_buffer_size));

  valid = true;
  return true;
}

bool CusparseLtPlan::compress(const __half* dA, cudaStream_t stream) {
  if (!valid || !dA || !d_valid || !d_compressed) return false;

  CHECK_LT(cusparseLtSpMMAPruneCheck(&handle, &matmulDesc, dA, d_valid, stream));

  size_t compressed_buffer_size = 0;
  cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compressed_buffer_size);
  void* d_compressed_buffer = (char*)d_compressed + compressed_size;

  CHECK_LT(cusparseLtSpMMACompress(&handle, &plan, dA, d_compressed, d_compressed_buffer, stream));

  return true;
}

bool CusparseLtPlan::matmul(const __half* dB, __half* dC, cudaStream_t stream) {
  if (!valid || !d_compressed || !dB || !dC) return false;

  float alpha = 1.0f, beta = 0.0f;

  CHECK_LT(cusparseLtMatmul(
      &handle, &plan,
      &alpha, d_compressed, dB,
      &beta, dC, dC,
      workspace, &stream, 1));

  return true;
}

} // namespace sparseflow
