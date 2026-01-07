#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace sparseflow {

enum class DType : uint8_t { F16 = 0, BF16 = 1, F32 = 2 };
enum class SparsityType : uint8_t { DENSE = 0, NM_2_4 = 1, NM_4_8 = 2 };
enum class Layout : uint8_t { ROW_MAJOR = 0, COL_MAJOR = 1 };

struct MatmulDesc {
  int M, N, K;
  DType dtype;
  SparsityType sparsity_type;
  float sparsity_ratio;
  Layout layout;
};

enum class KernelID : uint8_t {
  DENSE_TC_128 = 0,
  HYBRID_NM_32 = 1,
  SPARSE_NM_32 = 2,
  SPARSE_NM_64 = 3
};

struct BenchmarkResult {
  KernelID kernel_id;
  float mean_time_ms;
  float stddev_time_ms;
  float throughput_gflops;
  bool high_variance;
  bool used_cached_compression;
};

struct BenchmarkConfig {
  int warmup_iters = 3;
  int timed_iters = 10;
  float max_cv = 0.05f;
};

class MicroBenchmark {
public:
  explicit MicroBenchmark(const BenchmarkConfig& config = BenchmarkConfig());
  ~MicroBenchmark();

  std::vector<BenchmarkResult> benchmark_all(
      const std::vector<KernelID>& candidates,
      const MatmulDesc& desc,
      void* A, void* B, void* C,
      cudaStream_t stream);

  bool is_kernel_available(KernelID kernel);

private:
  BenchmarkResult benchmark_kernel(
      KernelID kernel,
      const MatmulDesc& desc,
      void* A, void* B, void* C,
      cudaStream_t stream);

  bool launch_kernel(
      KernelID kernel,
      const MatmulDesc& desc,
      void* A, void* B, void* C,
      cudaStream_t stream,
      bool* used_cached_compression = nullptr);

  BenchmarkConfig config_;
  cudaEvent_t start_, stop_;
  bool cuda_initialized_;
};

} // namespace sparseflow
