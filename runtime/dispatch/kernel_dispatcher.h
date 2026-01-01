#pragma once
#include "../cache/kernel_cache.h"
#include "../benchmark/micro_bench.h"

namespace sparseflow {

struct GPUInfo {
  int sm_arch;
  int num_sms;
  size_t shared_mem_kb;
};

class KernelDispatcher {
public:
  explicit KernelDispatcher(const std::string& cache_path = "kernel_cache.db");
  
  KernelID select_kernel(
    const MatmulDesc& desc,
    const GPUInfo& gpu_info,
    void* A, void* B, void* C,
    cudaStream_t stream = 0
  );

private:
  KernelCache cache_;
  MicroBenchmark benchmark_;
  
  KernelSelectionKey make_key(const MatmulDesc& desc, const GPUInfo& gpu);
  std::vector<KernelID> get_candidates(const MatmulDesc& desc);
  bool validate_sparsity_metadata(const MatmulDesc& desc);
};

} // namespace sparseflow
