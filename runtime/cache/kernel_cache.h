#pragma once
#include <string>
#include <cstdint>
#include "../benchmark/micro_bench.h"

namespace sparseflow {

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
  bool sparse_failed;  // NEW: Track if sparse attempt failed
  int failure_count;   // NEW: Number of failures for this key
};

class KernelCache {
public:
  explicit KernelCache(const std::string& db_path);
  ~KernelCache();

  bool lookup(const KernelSelectionKey& key, KernelSelectionValue* out_value) const;
  void insert(const KernelSelectionKey& key, const KernelSelectionValue& value);
  void record_failure(const KernelSelectionKey& key);  // NEW
  void clear();
  int32_t size() const;

private:
  struct Impl;
  Impl* impl_;
};

} // namespace sparseflow
