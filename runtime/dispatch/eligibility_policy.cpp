#include "eligibility_policy.h"
#include <unordered_map>
#include <mutex>
#include <sstream>

namespace sparseflow {

// Blacklist storage (key -> failure count)
static std::unordered_map<uint64_t, int> g_blacklist;
static std::mutex g_blacklist_mutex;

static uint64_t hash_key(const KernelSelectionKey& key) {
  uint64_t h = 0;
  h ^= ((uint64_t)key.M_bucket << 48);
  h ^= ((uint64_t)key.N_bucket << 32);
  h ^= ((uint64_t)key.K_bucket << 16);
  h ^= ((uint64_t)key.dtype << 8);
  h ^= ((uint64_t)key.sparsity_type);
  return h;
}

size_t EligibilityPolicy::compute_flops(const MatmulDesc& desc) {
  return (size_t)2 * desc.M * desc.N * desc.K;
}

EligibilityResult EligibilityPolicy::check_sparse_eligibility(
    const MatmulDesc& desc,
    const KernelSelectionKey& key,
    const KernelCache& cache,
    bool compression_cached) {
  
  EligibilityResult result;
  result.sparse_eligible = false;
  
  // Rule 1: Must be 2:4 sparsity
  if (desc.sparsity_type != SparsityType::NM_2_4) {
    result.reason = EligibilityReason::WRONG_SPARSITY_TYPE;
    result.message = "Only 2:4 sparsity supported";
    return result;
  }
  
  // Rule 2: Minimum size threshold
  if (compute_flops(desc) < MIN_SPARSE_FLOPS) {
    result.reason = EligibilityReason::MATRIX_TOO_SMALL;
    std::ostringstream oss;
    oss << "Matrix too small (" << desc.M << "x" << desc.N << "x" << desc.K << ")";
    result.message = oss.str();
    return result;
  }
  
  // Rule 3: Check blacklist
  if (is_blacklisted(key)) {
    result.reason = EligibilityReason::SPARSE_BLACKLISTED;
    result.message = "Sparse failed too many times for this shape";
    return result;
  }
  
  // Rule 4: If compression not cached, skip sparse (conservative)
  if (!compression_cached) {
    result.reason = EligibilityReason::CACHE_MISS_NO_COMPRESSION;
    result.message = "Compression not cached, skipping sparse";
    return result;
  }
  
  // All checks passed
  result.sparse_eligible = true;
  result.reason = EligibilityReason::ELIGIBLE;
  result.message = "Sparse eligible";
  return result;
}

void EligibilityPolicy::record_sparse_failure(const KernelSelectionKey& key) {
  std::lock_guard<std::mutex> lock(g_blacklist_mutex);
  uint64_t h = hash_key(key);
  g_blacklist[h]++;
}

bool EligibilityPolicy::is_blacklisted(const KernelSelectionKey& key) {
  std::lock_guard<std::mutex> lock(g_blacklist_mutex);
  uint64_t h = hash_key(key);
  auto it = g_blacklist.find(h);
  if (it == g_blacklist.end()) return false;
  return it->second >= MAX_FAILURES_BEFORE_BLACKLIST;
}

void EligibilityPolicy::clear_blacklist() {
  std::lock_guard<std::mutex> lock(g_blacklist_mutex);
  g_blacklist.clear();
}

} // namespace sparseflow
