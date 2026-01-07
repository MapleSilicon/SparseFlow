#include "kernel_key.h"
#include "../benchmark/micro_bench.h"
#include <sstream>

namespace sparseflow {

std::string KernelKey::to_string() const {
  std::ostringstream oss;
  oss << "M=" << M
      << ";N=" << N
      << ";K=" << K
      << ";dtype=" << dtype
      << ";layout=" << layout
      << ";sp=" << sparsity_type
      << ";sm=" << sm_arch;
  return oss.str();
}

KernelKey KernelKey::from(const MatmulDesc& d, const GPUInfo& gpu) {
  KernelKey k;
  k.M = d.M;
  k.N = d.N;
  k.K = d.K;
  k.dtype = static_cast<int>(d.dtype);
  k.layout = static_cast<int>(d.layout);
  k.sparsity_type = static_cast<int>(d.sparsity_type);
  k.sm_arch = gpu.sm_arch;
  return k;
}

} // namespace sparseflow
