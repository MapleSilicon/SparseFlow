#pragma once
#include "../benchmark/micro_bench.h"
#include "../cache/kernel_cache.h"
#include <vector>

namespace sparseflow {

enum class EligibilityReason {
  ELIGIBLE,
  WRONG_SPARSITY_TYPE,
  MATRIX_TOO_SMALL,
  SPARSE_BLACKLISTED,
  CACHE_MISS_NO_COMPRESSION,
  FORCED_DENSE
};

struct EligibilityResult {
  bool sparse_eligible;
  EligibilityReason reason;
  std::string message;
};

class EligibilityPolicy {
public:
  // Configuration
  static constexpr size_t MIN_SPARSE_FLOPS = 512 * 512 * 512;  // ~134M FLOPs
  static constexpr int MAX_FAILURES_BEFORE_BLACKLIST = 3;
  
  // Check if sparse kernel should be considered
  static EligibilityResult check_sparse_eligibility(
      const MatmulDesc& desc,
      const KernelSelectionKey& key,
      const KernelCache& cache,
      bool compression_cached);
  
  // Record a sparse failure for blacklisting
  static void record_sparse_failure(const KernelSelectionKey& key);
  
  // Check if sparse is blacklisted for this shape
  static bool is_blacklisted(const KernelSelectionKey& key);
  
  // Clear blacklist (for testing/reset)
  static void clear_blacklist();

private:
  static size_t compute_flops(const MatmulDesc& desc);
};

} // namespace sparseflow
