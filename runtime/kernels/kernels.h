#pragma once
#include "../benchmark/micro_bench.h"
#include <cuda_runtime.h>

namespace sparseflow {

bool launch_dense_tc_128(const MatmulDesc& desc,
                         void* A, void* B, void* C,
                         cudaStream_t stream);

bool launch_sparse_nm_32(const MatmulDesc& desc,
                         void* A, void* B, void* C,
                         cudaStream_t stream,
                         bool* used_cached_compression = nullptr);

} // namespace sparseflow
