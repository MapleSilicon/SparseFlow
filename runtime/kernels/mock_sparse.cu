#include <cuda_runtime.h>
#include <iostream>
#include "../benchmark/micro_bench.h"

namespace sparseflow {

// Mock sparse kernel - simulates 2:4 sparse performance
// Currently just calls dense kernel
// TODO: Replace with real cuSPARSELt or MLIR-generated sparse kernel
bool launch_sparse_nm_32(const MatmulDesc& desc,
                         void* A, void* B, void* C,
                         cudaStream_t stream) {
  
  // For now, just use dense kernel
  // In real implementation, this would use 2:4 sparse tensor cores
  extern bool launch_dense_tc_128(const MatmulDesc&, void*, void*, void*, cudaStream_t);
  
  return launch_dense_tc_128(desc, A, B, C, stream);
}

} // namespace sparseflow
