#pragma once
#include <cstdint>
#include <string>

namespace sparseflow {

struct KernelKey;
struct GPUInfo;

// Forward declare KernelID from micro_bench.h
enum class KernelID : uint8_t;

struct KernelCacheEntry {
  bool valid = false;
  KernelID best_kernel;
  double best_gflops = 0.0;
  
  // Phase-2B: sparse failure quarantine
  bool sparse_failed = false;
  int failure_count = 0;
  int last_failure_code = 0;
  int64_t last_failure_ts = 0;
};

// Legacy types (keep at bottom after new types)
struct KernelSelectionKey {
  uint16_t M_bucket, N_bucket, K_bucket;
  uint8_t dtype;
  uint8_t sparsity_type;
  uint8_t sm_arch;
  bool operator==(const KernelSelectionKey& other) const;
};

struct KernelSelectionValue {
  KernelID kernel_id;
  float throughput_gflops;
  bool sparse_failed = false;
  int failure_count = 0;
};

class KernelCache {
public:
  explicit KernelCache(const std::string& db_path);
  ~KernelCache();

  KernelCache(const KernelCache&) = delete;
  KernelCache& operator=(const KernelCache&) = delete;

  // New Phase-2B API
  bool get(const KernelKey& key, KernelCacheEntry* out) const;
  bool upsert_best(const KernelKey& key, KernelID best_kernel, double best_gflops);
  bool mark_sparse_failed(const KernelKey& key, int failure_code);
  bool clear_sparse_failed(const KernelKey& key);
  bool is_sparse_blacklisted(const KernelKey& key) const;
  
  // Legacy compatibility API
  bool lookup(const KernelSelectionKey& key, KernelSelectionValue* out_value) const;
  void insert(const KernelSelectionKey& key, const KernelSelectionValue& value);
  void record_failure(const KernelSelectionKey& key);
  void clear();
  int32_t size() const;

private:
  struct Impl;
  Impl* impl_;
};

} // namespace sparseflow
