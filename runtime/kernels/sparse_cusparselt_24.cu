#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include "../benchmark/micro_bench.h"
#include "../cache/compressed_cache.h"
#include "cusparselt_plan.h"

namespace sparseflow {

// Plan cache: one plan per shape
static std::unordered_map<LtPlanKey, CusparseLtPlan*, LtPlanKeyHash> g_plan_cache;
static std::mutex g_plan_mutex;

bool launch_sparse_nm_32(const MatmulDesc& desc,
                         void* A, void* B, void* C,
                         cudaStream_t stream,
                         bool* used_cached_compression) {
  
  if (desc.dtype != DType::F16) {
    return false;
  }

  __half* dA = static_cast<__half*>(A);
  __half* dB = static_cast<__half*>(B);
  __half* dC = static_cast<__half*>(C);

  // Build plan key
  LtPlanKey key;
  key.M = desc.M;
  key.N = desc.N;
  key.K = desc.K;
  key.dtype = static_cast<int>(desc.dtype);
  key.sparsity = static_cast<int>(desc.sparsity_type);

  // Check if plan exists
  CusparseLtPlan* plan = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    auto it = g_plan_cache.find(key);
    if (it == g_plan_cache.end()) {
      // Create new plan
      plan = new CusparseLtPlan();
      if (!plan->init(desc.M, desc.N, desc.K)) {
        delete plan;
        return false;
      }
      g_plan_cache[key] = plan;
    } else {
      plan = it->second;
    }
  }

  if (!plan || !plan->valid) {
    return false;
  }

  // Check if we need to compress
  // Use content hash to determine if this is cached data
  CompressedBufferKey cache_key;
  size_t A_bytes = (size_t)desc.M * desc.K * sizeof(__half);
  cache_key.content_hash = compute_matrix_hash(A, A_bytes);
  cache_key.M = desc.M;
  cache_key.N = desc.N;
  cache_key.K = desc.K;
  cache_key.dtype = static_cast<uint8_t>(desc.dtype);
  cache_key.sparsity_type = static_cast<uint8_t>(desc.sparsity_type);

  auto& compressed_cache = get_compressed_cache();
  bool is_cached = compressed_cache.contains(cache_key);

  if (used_cached_compression) {
    *used_cached_compression = is_cached;
  }

  // If not cached, compress now
  if (!is_cached) {
    if (!plan->compress(dA, stream)) {
      return false;
    }
    
    // Mark as cached (note: we're using plan's buffer, not storing separately)
    CompressedBufferValue dummy;
    dummy.d_compressed = plan->d_compressed;
    dummy.d_compressed_buffer = nullptr;
    dummy.compressed_size = 0;
    dummy.compressed_buffer_size = 0;
    compressed_cache.put(cache_key, std::move(dummy));
  }

  // Execute matmul
  if (!plan->matmul(dB, dC, stream)) {
    return false;
  }

  return true;
}

} // namespace sparseflow
