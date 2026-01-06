// src/runtime/sparse_runtime.h
// SparseFlow Runtime API

#ifndef SPARSEFLOW_RUNTIME_H
#define SPARSEFLOW_RUNTIME_H

#include "sparseflow/epilogue.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sparseflow {
namespace runtime {

// Execute sparse GEMM with epilogue
cudaError_t sparse_gemm_with_epilogue(
    const half* A,          // M × K dense
    const half* Bc,         // K × N compressed (50% size)
    const uint32_t* E,      // Metadata
    half* C,                // M × N output
    int M, int N, int K,
    const EpilogueConfig& epilogue,
    cudaStream_t stream = 0
);

// Helper: Get optimal launch configuration
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem_bytes;
};

LaunchConfig get_launch_config(int M, int N, int K);

} // namespace runtime
} // namespace sparseflow

#endif // SPARSEFLOW_RUNTIME_H
