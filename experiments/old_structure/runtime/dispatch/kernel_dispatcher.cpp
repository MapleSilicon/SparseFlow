#include "kernel_dispatcher.h"
#include "eligibility_policy.h"
#include "../cache/compressed_cache.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

namespace sparseflow {

KernelDispatcher::KernelDispatcher(const std::string& cache_path)
    : cache_(cache_path), benchmark_() {}

static uint16_t bucket_dimension(int dim) {
  if (dim <= 0) return 0;
  uint32_t x = (dim <= 32) ? 32u : (uint32_t)dim;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  if (x > 65535u) x = 65535u;
  return (uint16_t)x;
}

std::vector<KernelID> KernelDispatcher::get_candidates(const MatmulDesc& desc) {
  std::vector<KernelID> candidates;
  candidates.push_back(KernelID::DENSE_TC_128);

  if (desc.sparsity_type == SparsityType::NM_2_4 &&
      desc.sparsity_ratio >= 0.4f && desc.sparsity_ratio <= 0.6f) {
    candidates.push_back(KernelID::SPARSE_NM_32);
  }

  return candidates;
}

KernelID KernelDispatcher::select_kernel(
    const MatmulDesc& desc,
    const GPUInfo& gpu,
    void* A, void* B, void* C,
    cudaStream_t stream) {
  
  KernelSelectionKey key;
  key.M_bucket = bucket_dimension(desc.M);
  key.N_bucket = bucket_dimension(desc.N);
  key.K_bucket = bucket_dimension(desc.K);
  key.dtype = static_cast<uint8_t>(desc.dtype);
  key.sparsity_type = static_cast<uint8_t>(desc.sparsity_type);
  key.sm_arch = gpu.sm_arch;
  
  // Check cache first
  KernelSelectionValue cached_value;
  if (cache_.lookup(key, &cached_value)) {
    std::cout << "[Cache HIT] " << static_cast<int>(cached_value.kernel_id)
              << " (" << cached_value.throughput_gflops << " GFLOPS)" << std::endl;
    return cached_value.kernel_id;
  }
  
  std::cout << "[Cache MISS] Evaluating kernels..." << std::endl;
  
  // Check if compression is cached (for eligibility)
  bool compression_cached = false;
  if (A && desc.sparsity_type == SparsityType::NM_2_4) {
    CompressedBufferKey ckey;
    size_t A_bytes = (size_t)desc.M * desc.K * 
                     ((desc.dtype == DType::F16) ? 2 : 4);
    ckey.content_hash = compute_matrix_hash(A, A_bytes);
    ckey.M = desc.M;
    ckey.N = desc.N;
    ckey.K = desc.K;
    ckey.dtype = static_cast<uint8_t>(desc.dtype);
    ckey.sparsity_type = static_cast<uint8_t>(desc.sparsity_type);
    
    compression_cached = get_compressed_cache().contains(ckey);
  }
  
  // Check sparse eligibility
  auto eligibility = EligibilityPolicy::check_sparse_eligibility(
      desc, key, cache_, compression_cached);
  
  if (!eligibility.sparse_eligible) {
    std::cout << "[Policy] Sparse skipped: " << eligibility.message << std::endl;
    std::cout << "[Policy] Using DENSE" << std::endl;
    
    KernelSelectionValue value;
    value.kernel_id = KernelID::DENSE_TC_128;
    value.throughput_gflops = 0.0f;  // Will be measured later
    value.sparse_failed = false;
    value.failure_count = 0;
    cache_.insert(key, value);
    
    return KernelID::DENSE_TC_128;
  }
  
  // Sparse is eligible - benchmark both
  std::cout << "[Policy] Sparse eligible, benchmarking..." << std::endl;
  
  auto candidates = get_candidates(desc);
  
  // Allocate buffers if needed
  void *A_tmp = A, *B_tmp = B, *C_tmp = C;
  bool allocated_tmp = false;
  
  if (!A || !B || !C) {
    size_t dtype_size = (desc.dtype == DType::F16) ? sizeof(__half) : sizeof(float);
    size_t A_bytes = (size_t)desc.M * desc.K * dtype_size;
    size_t B_bytes = (size_t)desc.K * desc.N * dtype_size;
    size_t C_bytes = (size_t)desc.M * desc.N * dtype_size;
    
    cudaMalloc(&A_tmp, A_bytes);
    cudaMalloc(&B_tmp, B_bytes);
    cudaMalloc(&C_tmp, C_bytes);
    
    if (desc.dtype == DType::F16) {
      std::vector<__half> hA((size_t)desc.M * desc.K);
      for (size_t i = 0; i < hA.size(); i += 4) {
        hA[i]   = __float2half(1.0f);
        hA[i+1] = __float2half(1.0f);
        hA[i+2] = __float2half(0.0f);
        hA[i+3] = __float2half(0.0f);
      }
      cudaMemcpy(A_tmp, hA.data(), A_bytes, cudaMemcpyHostToDevice);
    }
    
    cudaMemset(B_tmp, 0x3C, B_bytes);
    cudaMemset(C_tmp, 0, C_bytes);
    allocated_tmp = true;
  }
  
  auto results = benchmark_.benchmark_all(candidates, desc, A_tmp, B_tmp, C_tmp, stream);
  
  if (allocated_tmp) {
    cudaFree(A_tmp);
    cudaFree(B_tmp);
    cudaFree(C_tmp);
  }
  
  if (results.empty()) {
    std::cerr << "[ERROR] No valid results" << std::endl;
    return KernelID::DENSE_TC_128;
  }
  
  // Find best valid kernel
  BenchmarkResult* best = nullptr;
  bool sparse_failed = false;
  
  for (auto& r : results) {
    if (r.mean_time_ms <= 0) {
      if (r.kernel_id == KernelID::SPARSE_NM_32) {
        sparse_failed = true;
        std::cout << "[WARNING] Sparse kernel failed" << std::endl;
      }
      continue;
    }
    if (!best || r.throughput_gflops > best->throughput_gflops) {
      best = &r;
    }
  }
  
  if (!best) {
    std::cerr << "[ERROR] No valid kernel found" << std::endl;
    return KernelID::DENSE_TC_128;
  }
  
  // Record failure if sparse failed
  if (sparse_failed) {
    EligibilityPolicy::record_sparse_failure(key);
    cache_.record_failure(key);
  }
  
  std::cout << "[Selected] " << static_cast<int>(best->kernel_id)
            << " (" << best->throughput_gflops << " GFLOPS)" << std::endl;
  
  // Cache result
  KernelSelectionValue value;
  value.kernel_id = best->kernel_id;
  value.throughput_gflops = best->throughput_gflops;
  value.sparse_failed = sparse_failed;
  value.failure_count = sparse_failed ? 1 : 0;
  cache_.insert(key, value);
  
  return best->kernel_id;
}

} // namespace sparseflow
