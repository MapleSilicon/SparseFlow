#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <cstddef>
#include <cstdint>

namespace sparseflow {

struct LtPlanKey {
  int M = 0, N = 0, K = 0;
  int dtype = 0;
  int sparsity = 0;

  bool operator==(const LtPlanKey& o) const {
    return M == o.M && N == o.N && K == o.K && dtype == o.dtype && sparsity == o.sparsity;
  }
};

struct LtPlanKeyHash {
  size_t operator()(const LtPlanKey& k) const {
    size_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    mix((uint64_t)(uint32_t)k.M);
    mix((uint64_t)(uint32_t)k.N);
    mix((uint64_t)(uint32_t)k.K);
    mix((uint64_t)(uint32_t)k.dtype);
    mix((uint64_t)(uint32_t)k.sparsity);
    return h;
  }
};

struct CusparseLtPlan {
  bool valid = false;

  int M = 0, N = 0, K = 0;

  cusparseLtHandle_t handle{};

  cusparseLtMatDescriptor_t matA{};
  cusparseLtMatDescriptor_t matB{};
  cusparseLtMatDescriptor_t matC{};

  cusparseLtMatmulDescriptor_t matmulDesc{};
  cusparseLtMatmulAlgSelection_t algSel{};
  cusparseLtMatmulPlan_t plan{};

  int* d_valid = nullptr;

  void* d_compressed = nullptr;
  size_t compressed_size = 0;

  void* workspace = nullptr;
  size_t workspace_size = 0;

  CusparseLtPlan() = default;
  ~CusparseLtPlan();

  bool init(int M_, int N_, int K_);
  bool compress(const __half* dA, cudaStream_t stream);
  bool matmul(const __half* dB, __half* dC, cudaStream_t stream);
};

} // namespace sparseflow
