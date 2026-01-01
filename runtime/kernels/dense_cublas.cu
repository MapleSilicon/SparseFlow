#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../benchmark/micro_bench.h"

namespace sparseflow {

static cublasHandle_t g_cublas_handle = nullptr;

static bool ensure_cublas_handle() {
  if (!g_cublas_handle) {
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "[cuBLAS] Failed to create handle (status=" << status << ")" << std::endl;
      return false;
    }
  }
  return true;
}

static cudaDataType_t to_cuda_dtype(DType dt) {
  switch (dt) {
    case DType::F16:  return CUDA_R_16F;
    case DType::BF16: return CUDA_R_16BF;
    case DType::F32:  return CUDA_R_32F;
    default:          return CUDA_R_16F;
  }
}

bool launch_dense_tc_128(const MatmulDesc& desc,
                         void* A, void* B, void* C,
                         cudaStream_t stream) {
  if (!ensure_cublas_handle()) {
    return false;
  }
  
  cublasSetStream(g_cublas_handle, stream);
  
  const float alpha = 1.0f;
  const float beta = 0.0f;
  
  cudaDataType_t dtype = to_cuda_dtype(desc.dtype);
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  
  int m = desc.N;
  int n = desc.M;
  int k = desc.K;
  
  int lda = desc.N;
  int ldb = desc.K;
  int ldc = desc.N;
  
  cublasStatus_t status = cublasGemmEx(
    g_cublas_handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    B, dtype, lda,
    A, dtype, ldb,
    &beta,
    C, dtype, ldc,
    compute_type,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
  );
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "[cuBLAS] GemmEx failed (status=" << status << ")" << std::endl;
    return false;
  }
  
  return true;
}

} // namespace sparseflow
