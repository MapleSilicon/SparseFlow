#pragma once
#include <cstdint>
#include <string>

namespace sparseflow {

struct MatmulDesc;

// Match the actual struct name from benchmark
struct GPUInfo {
  int sm_arch;
};

struct KernelKey {
  int M=0, N=0, K=0;
  int dtype=0;
  int layout=0;
  int sparsity_type=0;
  int sm_arch=0;

  std::string to_string() const;
  static KernelKey from(const MatmulDesc& d, const GPUInfo& gpu);
};

} // namespace sparseflow
