#include "micro_bench.h"
#include "../kernels/kernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace sparseflow {

static constexpr bool DENSE_TC_128_AVAILABLE = true;
static constexpr bool HYBRID_NM_32_AVAILABLE = false;
static constexpr bool SPARSE_NM_32_AVAILABLE = true;
static constexpr bool SPARSE_NM_64_AVAILABLE = false;

MicroBenchmark::MicroBenchmark(const BenchmarkConfig& config)
    : config_(config), cuda_initialized_(false) {
  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
}

MicroBenchmark::~MicroBenchmark() {
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

bool MicroBenchmark::is_kernel_available(KernelID kernel) {
  switch (kernel) {
    case KernelID::DENSE_TC_128: return DENSE_TC_128_AVAILABLE;
    case KernelID::HYBRID_NM_32: return HYBRID_NM_32_AVAILABLE;
    case KernelID::SPARSE_NM_32: return SPARSE_NM_32_AVAILABLE;
    case KernelID::SPARSE_NM_64: return SPARSE_NM_64_AVAILABLE;
  }
  return false;
}

static void flush_l2_cache() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  size_t l2_size = prop.l2CacheSize;
  if (l2_size > 0) {
    void* flush_buffer;
    cudaMalloc(&flush_buffer, l2_size);
    cudaMemset(flush_buffer, 0, l2_size);
    cudaFree(flush_buffer);
  }
}

bool MicroBenchmark::launch_kernel(
    KernelID kernel,
    const MatmulDesc& desc,
    void* A, void* B, void* C,
    cudaStream_t stream,
    bool* used_cached_compression) {
  
  switch (kernel) {
    case KernelID::DENSE_TC_128:
      return launch_dense_tc_128(desc, A, B, C, stream);
    case KernelID::HYBRID_NM_32:
      return false;
    case KernelID::SPARSE_NM_32:
      return launch_sparse_nm_32(desc, A, B, C, stream, used_cached_compression);
    case KernelID::SPARSE_NM_64:
      return false;
  }
  return false;
}

BenchmarkResult MicroBenchmark::benchmark_kernel(
    KernelID kernel,
    const MatmulDesc& desc,
    void* A, void* B, void* C,
    cudaStream_t stream) {
  
  std::vector<float> times;
  times.reserve(config_.timed_iters);
  bool retried = false;
  int current_iters = config_.timed_iters;
  
  flush_l2_cache();

  // Check if sparse kernel will use cached compression
  bool used_cache = false;
  
retry:
  times.clear();
  
  for (int i = 0; i < config_.warmup_iters; ++i) {
    if (!launch_kernel(kernel, desc, A, B, C, stream, &used_cache)) {
      BenchmarkResult invalid;
      invalid.kernel_id = kernel;
      invalid.mean_time_ms = -1.0f;
      invalid.throughput_gflops = 0.0f;
      invalid.high_variance = true;
      invalid.used_cached_compression = false;
      return invalid;
    }
  }
  cudaStreamSynchronize(stream);

  for (int i = 0; i < current_iters; ++i) {
    cudaEventRecord(start_, stream);
    
    bool iter_used_cache = false;
    if (!launch_kernel(kernel, desc, A, B, C, stream, &iter_used_cache)) {
      BenchmarkResult invalid;
      invalid.kernel_id = kernel;
      invalid.mean_time_ms = -1.0f;
      invalid.throughput_gflops = 0.0f;
      invalid.high_variance = true;
      invalid.used_cached_compression = false;
      return invalid;
    }
    
    used_cache = iter_used_cache;
    
    cudaEventRecord(stop_, stream);
    cudaStreamSynchronize(stream);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    times.push_back(ms);
  }

  // 🔒 CRITICAL: Reject sparse timing if compression not cached
  if (kernel == KernelID::SPARSE_NM_32 && !used_cache) {
    BenchmarkResult invalid;
    invalid.kernel_id = kernel;
    invalid.mean_time_ms = -1.0f;
    invalid.throughput_gflops = 0.0f;
    invalid.high_variance = true;
    invalid.used_cached_compression = false;
    return invalid;
  }

  float mean_time = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
  
  float variance = 0.0f;
  for (float t : times) {
    float diff = t - mean_time;
    variance += diff * diff;
  }
  variance /= times.size();
  float stddev = std::sqrt(variance);
  float cv = stddev / mean_time;

  if (cv > config_.max_cv && !retried) {
    retried = true;
    current_iters = config_.timed_iters * 2;
    goto retry;
  }

  BenchmarkResult result;
  result.kernel_id = kernel;
  result.mean_time_ms = mean_time;
  result.stddev_time_ms = stddev;
  result.high_variance = (cv > config_.max_cv);
  result.used_cached_compression = used_cache;

  int64_t ops = 2LL * desc.M * desc.N * desc.K;
  bool kernel_is_sparse =
      (kernel == KernelID::SPARSE_NM_32) || (kernel == KernelID::SPARSE_NM_64);
  
  if (kernel_is_sparse && desc.sparsity_type == SparsityType::NM_2_4) {
    ops /= 2;
  }
  
  float time_s = mean_time / 1000.0f;
  result.throughput_gflops = static_cast<float>(ops) / (time_s * 1e9f);

  return result;
}

std::vector<BenchmarkResult> MicroBenchmark::benchmark_all(
    const std::vector<KernelID>& candidates,
    const MatmulDesc& desc,
    void* A, void* B, void* C,
    cudaStream_t stream) {
  
  if (!cuda_initialized_) {
    std::cout << "[Benchmark] CUDA context initialized" << std::endl;
    cuda_initialized_ = true;
  }

  int implemented_count = 0;
  for (KernelID k : candidates) {
    if (is_kernel_available(k)) {
      implemented_count++;
    }
  }

  std::cout << "[Benchmark] " << implemented_count << " of " 
            << candidates.size() << " kernels implemented" << std::endl;

  std::vector<BenchmarkResult> results;
  int kernel_idx = 0;
  
  for (KernelID kernel : candidates) {
    if (!is_kernel_available(kernel)) {
      kernel_idx++;
      continue;
    }

    std::cout << "[Benchmark] Testing kernel " << kernel_idx;
    switch(kernel) {
      case KernelID::DENSE_TC_128:
        std::cout << " (DENSE)";
        break;
      case KernelID::SPARSE_NM_32:
        std::cout << " (SPARSE 2:4)";
        break;
      default:
        break;
    }
    std::cout << "..." << std::endl;
    
    auto result = benchmark_kernel(kernel, desc, A, B, C, stream);
    
    if (result.mean_time_ms > 0) {
      results.push_back(result);
      std::cout << "[Benchmark]   → " << result.throughput_gflops << " GFLOPS";
      if (result.used_cached_compression) {
        std::cout << " (CACHED)";
      }
      std::cout << std::endl;
    } else {
      std::cout << "[Benchmark]   → INVALID (compression not cached)" << std::endl;
    }
    
    kernel_idx++;
  }

  return results;
}

} // namespace sparseflow
