#include "kernel_dispatcher.h"
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
    candidates.push_back(KernelID::HYBRID_NM_32);
    candidates.push_back(KernelID::SPARSE_NM_32);
    candidates.push_back(KernelID::SPARSE_NM_64);
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
  
  KernelSelectionValue cached_value;
  if (cache_.lookup(key, &cached_value)) {
    bool sparse_requested = (desc.sparsity_type == SparsityType::NM_2_4 &&
                            desc.sparsity_ratio >= 0.4f && desc.sparsity_ratio <= 0.6f);
    
    // 🔥 PROMOTION: If dense cached but sparse requested, re-check
    if (sparse_requested && cached_value.kernel_id == KernelID::DENSE_TC_128) {
      std::cout << "[Cache HIT] Dense cached, checking sparse promotion...\n";
      
      // Allocate buffers
      void *A_tmp = A, *B_tmp = B, *C_tmp = C;
      bool allocated_tmp = false;
      
      if (!A_tmp || !B_tmp || !C_tmp) {
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
      
      // Benchmark both
      std::vector<KernelID> promo = {KernelID::DENSE_TC_128, KernelID::SPARSE_NM_32};
      auto promo_results = benchmark_.benchmark_all(promo, desc, A_tmp, B_tmp, C_tmp, stream);
      
      // Pick best
      BenchmarkResult* best = nullptr;
      for (auto& r : promo_results) {
        if (r.mean_time_ms <= 0) continue;
        if (!best || r.throughput_gflops > best->throughput_gflops) best = &r;
      }
      
      if (best && best->kernel_id == KernelID::SPARSE_NM_32) {
        KernelSelectionValue nv;
        nv.kernel_id = best->kernel_id;
        nv.throughput_gflops = best->throughput_gflops;
        cache_.insert(key, nv);
        
        std::cout << "[Cache PROMOTE] Sparse now selected (" << nv.throughput_gflops << " GFLOPS)\n";
        
        if (allocated_tmp) {
          cudaFree(A_tmp);
          cudaFree(B_tmp);
          cudaFree(C_tmp);
        }
        return nv.kernel_id;
      }
      
      if (allocated_tmp) {
        cudaFree(A_tmp);
        cudaFree(B_tmp);
        cudaFree(C_tmp);
      }
    }
    
    std::cout << "[Cache HIT] Using kernel " << static_cast<int>(cached_value.kernel_id)
              << " (" << cached_value.throughput_gflops << " GFLOPS)" << std::endl;
    return cached_value.kernel_id;
  }
  
  std::cout << "[Cache MISS] Running micro-benchmark..." << std::endl;
  
  auto candidates = get_candidates(desc);
  std::cout << "  Candidates: " << candidates.size() << " kernels" << std::endl;
  
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
    std::cerr << "[Dispatcher] ERROR: No valid benchmark results" << std::endl;
    return KernelID::DENSE_TC_128;
  }
  
  // Pick best
  BenchmarkResult* best_result = nullptr;
  for (auto& r : results) {
    if (!best_result || r.throughput_gflops > best_result->throughput_gflops) {
      best_result = &r;
    }
  }
  
  if (!best_result) {
    std::cerr << "[Dispatcher] ERROR: Could not select kernel" << std::endl;
    return KernelID::DENSE_TC_128;
  }
  
  std::cout << "  Selected: kernel " << static_cast<int>(best_result->kernel_id)
            << " (" << best_result->throughput_gflops << " GFLOPS)" << std::endl;
  
  KernelSelectionValue value;
  value.kernel_id = best_result->kernel_id;
  value.throughput_gflops = best_result->throughput_gflops;
  cache_.insert(key, value);
  
  return best_result->kernel_id;
}

} // namespace sparseflow
