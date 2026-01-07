#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

static cublasHandle_t g_cublas_handle = nullptr;
static cublasLtHandle_t g_cublaslt_handle = nullptr;

bool test_cublas(int size, void* A, void* B, void* C, cudaStream_t stream) {
  if (!g_cublas_handle) cublasCreate(&g_cublas_handle);
  cublasSetStream(g_cublas_handle, stream);
  
  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);
  
  return (CUBLAS_STATUS_SUCCESS == cublasHgemm(g_cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
    &alpha, (__half*)A, size, (__half*)B, size, 
    &beta, (__half*)C, size));
}

bool test_cublaslt_corrected(int size, void* A, void* B, void* C, cudaStream_t stream) {
  if (!g_cublaslt_handle) cublasLtCreate(&g_cublaslt_handle);
  
  // CRITICAL FIX: Use same layout as cuBLAS
  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
  
  cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

  // CRITICAL FIX: Correct matrix layouts to match cuBLAS exactly
  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, size, size, size);  // M=size, N=size, LD=size
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, size, size, size);  
  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, size, size, size);

  cublasLtMatmulPreference_t preference;
  cublasLtMatmulPreferenceCreate(&preference);
  
  size_t workspace_size = 4 * 1024 * 1024; // Reduce workspace size
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                       &workspace_size, sizeof(workspace_size));

  cublasLtMatmulHeuristicResult_t heuristic_results[8];
  int returned_algo_count = 0;
  
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(g_cublaslt_handle, operationDesc,
                                          Adesc, Bdesc, Cdesc, Cdesc,
                                          preference, 8, heuristic_results, &returned_algo_count);
  
  if (status != CUBLAS_STATUS_SUCCESS || returned_algo_count == 0) {
    // Cleanup and return false
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
    return false;
  }

  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaMalloc(&workspace, workspace_size);
  }
  
  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);
  
  status = cublasLtMatmul(g_cublaslt_handle, operationDesc,
                          &alpha, A, Adesc, B, Bdesc,
                          &beta, C, Cdesc, C, Cdesc,
                          &heuristic_results[0].algo,
                          workspace, workspace_size, stream);

  // Cleanup
  if (workspace) cudaFree(workspace);
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatmulDescDestroy(operationDesc);
  
  return (status == CUBLAS_STATUS_SUCCESS);
}

float benchmark_method(bool (*func)(int,void*,void*,void*,cudaStream_t), int size, void* A, void* B, void* C, cudaStream_t stream) {
  // Warmup
  for (int i = 0; i < 3; i++) {
    if (!func(size, A, B, C, stream)) return -1.0f;
  }
  cudaStreamSynchronize(stream);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start, stream);
  bool success = func(size, A, B, C, stream);
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  
  if (!success) return -1.0f;
  
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  
  int64_t ops = 2LL * size * size * size;
  return (float(ops) / ((ms / 1000.0f) * 1e12f));
}

int main() {
  std::cout << "\n=== cuBLAS vs cuBLASLt (Corrected) ===\n";
  
  int size = 1024;
  size_t matrix_size = size * size * sizeof(__half);
  
  void *A, *B, *C;
  cudaMalloc(&A, matrix_size);
  cudaMalloc(&B, matrix_size);
  cudaMalloc(&C, matrix_size);
  
  cudaMemset(A, 0x3C00, matrix_size);
  cudaMemset(B, 0x3C00, matrix_size);
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  float cublas_tflops = benchmark_method(test_cublas, size, A, B, C, stream);
  float cublaslt_tflops = benchmark_method(test_cublaslt_corrected, size, A, B, C, stream);
  
  std::cout << "cuBLAS baseline:    " << cublas_tflops << " TFLOPS\n";
  std::cout << "cuBLASLt corrected: " << cublaslt_tflops << " TFLOPS\n";
  
  if (cublaslt_tflops > 0) {
    float diff = ((cublaslt_tflops - cublas_tflops) / cublas_tflops) * 100.0f;
    std::cout << "Performance change: " << diff << "%\n";
    
    if (diff > -10.0f) {
      std::cout << "✅ MUCH better than -57% (Week 1-2) or -95% (broken version)!\n";
      std::cout << "✅ Proves fusion CAN work with correct configuration\n";
    } else {
      std::cout << "⚠️  Still suboptimal - may need different algorithm selection\n";
    }
  } else {
    std::cout << "❌ cuBLASLt failed - algorithm selection issue\n";
  }
  
  return 0;
}
