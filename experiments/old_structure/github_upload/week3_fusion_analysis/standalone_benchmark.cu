#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

using Clock = std::chrono::high_resolution_clock;

// Embed the cuBLAS function directly
static cublasHandle_t g_cublas_handle = nullptr;

static bool ensure_cublas_handle() {
  if (!g_cublas_handle) {
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    return (status == CUBLAS_STATUS_SUCCESS);
  }
  return true;
}

bool test_cublas(int M, int N, int K, void* A, void* B, void* C, cudaStream_t stream) {
  if (!ensure_cublas_handle()) return false;
  cublasSetStream(g_cublas_handle, stream);
  
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t status = cublasGemmEx(
    g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K, &alpha,
    B, CUDA_R_16F, N,
    A, CUDA_R_16F, K,
    &beta, C, CUDA_R_16F, N,
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  
  return (status == CUBLAS_STATUS_SUCCESS);
}

// Embed the cuBLASLt function directly  
static cublasLtHandle_t g_cublaslt_handle = nullptr;

bool test_cublaslt(int M, int N, int K, void* A, void* B, void* C, cudaStream_t stream) {
  if (!g_cublaslt_handle) {
    cublasLtCreate(&g_cublaslt_handle);
  }
  
  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_16F);
  
  cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, M, K, M);
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, N, K);  
  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, M);

  cublasLtMatmulPreference_t preference;
  cublasLtMatmulPreferenceCreate(&preference);
  size_t workspace_size = 32 * 1024 * 1024;
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));

  cublasLtMatmulHeuristicResult_t heuristic_results[4];
  int returned_algo_count = 0;
  cublasLtMatmulAlgoGetHeuristic(g_cublaslt_handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 4, heuristic_results, &returned_algo_count);
  
  if (returned_algo_count == 0) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc); 
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
    return false;
  }

  void* workspace;
  cudaMalloc(&workspace, workspace_size);
  
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t status = cublasLtMatmul(g_cublaslt_handle, operationDesc, &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc, C, Cdesc, &heuristic_results[0].algo, workspace, workspace_size, stream);

  cudaFree(workspace);
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc); 
  cublasLtMatmulDescDestroy(operationDesc);
  
  return (status == CUBLAS_STATUS_SUCCESS);
}

float benchmark_method(bool (*func)(int,int,int,void*,void*,void*,cudaStream_t), int size, void* A, void* B, void* C, cudaStream_t stream, const char* name) {
  std::cout << "[Benchmark] Testing " << name << "..." << std::endl;
  
  // Warmup
  for (int i = 0; i < 3; i++) {
    if (!func(size, size, size, A, B, C, stream)) {
      std::cout << "[ERROR] " << name << " failed!" << std::endl;
      return -1.0f;
    }
  }
  cudaStreamSynchronize(stream);
  
  // Timed runs
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::vector<float> times;
  for (int i = 0; i < 10; i++) {
    cudaEventRecord(start, stream);
    func(size, size, size, A, B, C, stream);
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    times.push_back(ms);
  }
  
  float mean_time = 0;
  for (float t : times) mean_time += t;
  mean_time /= times.size();
  
  int64_t ops = 2LL * size * size * size;
  float tflops = (float(ops) / ((mean_time / 1000.0f) * 1e12f));
  
  std::cout << "[Benchmark]   → " << std::fixed << std::setprecision(2) << tflops << " TFLOPS" << std::endl;
  return tflops;
}

int main() {
  std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
  std::cout << "║         Week-3 cuBLAS vs cuBLASLt Test            ║\n";
  std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

  for (int size : {1024, 2048}) {
    std::cout << "=== Testing " << size << "³ ===\n";
    
    size_t matrix_size = size * size * sizeof(__half);
    void *A, *B, *C;
    cudaMalloc(&A, matrix_size);
    cudaMalloc(&B, matrix_size);  
    cudaMalloc(&C, matrix_size);
    
    cudaMemset(A, 1, matrix_size);
    cudaMemset(B, 1, matrix_size);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    float cublas_tflops = benchmark_method(test_cublas, size, A, B, C, stream, "cuBLAS (baseline)");
    float cublaslt_tflops = benchmark_method(test_cublaslt, size, A, B, C, stream, "cuBLASLt (heuristic)");
    
    if (cublas_tflops > 0 && cublaslt_tflops > 0) {
      float improvement = ((cublaslt_tflops - cublas_tflops) / cublas_tflops) * 100.0f;
      std::cout << "\n📊 Results:\n";
      std::cout << "   cuBLAS:   " << cublas_tflops << " TFLOPS\n";
      std::cout << "   cuBLASLt: " << cublaslt_tflops << " TFLOPS (" << improvement << "%)\n";
      
      if (improvement > -10.0f) {
        std::cout << "✅ SUCCESS: Much better than your Week 1-2 result (-57%)!\n";
      }
    }
    
    cudaFree(A); cudaFree(B); cudaFree(C);
    cudaStreamDestroy(stream);
    std::cout << "\n";
  }
  
  std::cout << "Week-3 Analysis: cuBLASLt with proper heuristics should fix your -57% issue!\n";
  return 0;
}
