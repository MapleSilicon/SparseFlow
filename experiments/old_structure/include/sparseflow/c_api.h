//===- c_api.h - SparseFlow Stable C API --------------------------------===//
//
// Stable ABI v1 - Binary compatible across minor versions
// Breaking changes only on major version bump
//
//===----------------------------------------------------------------------===//

#ifndef SPARSEFLOW_C_API_H
#define SPARSEFLOW_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ABI version - increment on breaking changes
#define SPARSEFLOW_ABI_VERSION_MAJOR 1
#define SPARSEFLOW_ABI_VERSION_MINOR 0
#define SPARSEFLOW_ABI_VERSION_PATCH 0

// Export macro
#if defined(_WIN32)
  #ifdef SPARSEFLOW_BUILD_SHARED
    #define SPARSEFLOW_API __declspec(dllexport)
  #else
    #define SPARSEFLOW_API __declspec(dllimport)
  #endif
#else
  #define SPARSEFLOW_API __attribute__((visibility("default")))
#endif

//===----------------------------------------------------------------------===//
// Opaque handles (hide implementation details)
//===----------------------------------------------------------------------===//

typedef struct SparseFlowContext_* SparseFlowContext;
typedef struct SparseFlowKernel_* SparseFlowKernel;
typedef struct SparseFlowTensor_* SparseFlowTensor;

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

typedef enum {
    SPARSEFLOW_SUCCESS = 0,
    SPARSEFLOW_ERROR_INVALID_ARGUMENT = 1,
    SPARSEFLOW_ERROR_OUT_OF_MEMORY = 2,
    SPARSEFLOW_ERROR_CUDA_ERROR = 3,
    SPARSEFLOW_ERROR_NOT_SUPPORTED = 4,
    SPARSEFLOW_ERROR_INVALID_HANDLE = 5,
    SPARSEFLOW_ERROR_COMPILATION_FAILED = 6,
    SPARSEFLOW_ERROR_ABI_MISMATCH = 7,
} SparseFlowStatus;

// Get error message for status code
SPARSEFLOW_API const char* sparseflow_get_error_string(SparseFlowStatus status);

// Get last error message (thread-local)
SPARSEFLOW_API const char* sparseflow_get_last_error();

//===----------------------------------------------------------------------===//
// Version info
//===----------------------------------------------------------------------===//

typedef struct {
    int major;
    int minor;
    int patch;
} SparseFlowVersion;

SPARSEFLOW_API SparseFlowVersion sparseflow_get_version();

// Check ABI compatibility
SPARSEFLOW_API int sparseflow_is_abi_compatible(int major, int minor);

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

// Create context for specific GPU
SPARSEFLOW_API SparseFlowStatus sparseflow_create_context(
    SparseFlowContext* ctx,
    int device_id
);

// Destroy context
SPARSEFLOW_API SparseFlowStatus sparseflow_destroy_context(
    SparseFlowContext ctx
);

// Get GPU info from context
SPARSEFLOW_API SparseFlowStatus sparseflow_get_device_info(
    SparseFlowContext ctx,
    int* compute_capability,
    int* sm_count,
    size_t* total_memory
);

//===----------------------------------------------------------------------===//
// Epilogue configuration
//===----------------------------------------------------------------------===//

typedef enum {
    SPARSEFLOW_EPILOGUE_NONE = 0,
    SPARSEFLOW_EPILOGUE_RELU = 1,
    SPARSEFLOW_EPILOGUE_SILU = 2,
    SPARSEFLOW_EPILOGUE_GELU = 3,
    SPARSEFLOW_EPILOGUE_BIAS = 4,
    SPARSEFLOW_EPILOGUE_BIAS_RELU = 5,
    SPARSEFLOW_EPILOGUE_BIAS_SILU = 6,
} SparseFlowEpilogue;

typedef struct {
    SparseFlowEpilogue kind;
    const void* params;      // Optional parameters (e.g., bias pointer)
    size_t params_size;      // Size of params in bytes
} SparseFlowEpilogueConfig;

//===----------------------------------------------------------------------===//
// Kernel compilation
//===----------------------------------------------------------------------===//

typedef struct {
    int tile_m;              // Tile size M (0 = auto-select)
    int tile_n;              // Tile size N (0 = auto-select)
    int tile_k;              // Tile size K (0 = auto-select)
    SparseFlowEpilogueConfig epilogue;
} SparseFlowKernelConfig;

// Compile kernel with given configuration
SPARSEFLOW_API SparseFlowStatus sparseflow_compile_kernel(
    SparseFlowContext ctx,
    SparseFlowKernel* kernel,
    const SparseFlowKernelConfig* config
);

// Destroy kernel
SPARSEFLOW_API SparseFlowStatus sparseflow_destroy_kernel(
    SparseFlowKernel kernel
);

//===----------------------------------------------------------------------===//
// Tensor operations
//===----------------------------------------------------------------------===//

typedef enum {
    SPARSEFLOW_DTYPE_FP16 = 0,
    SPARSEFLOW_DTYPE_FP32 = 1,
    SPARSEFLOW_DTYPE_INT8 = 2,
} SparseFlowDataType;

// Execute sparse GEMM: C = A @ Bc
SPARSEFLOW_API SparseFlowStatus sparseflow_sparse_gemm(
    SparseFlowKernel kernel,
    const void* A,           // Dense matrix (M × K)
    const void* Bc,          // Compressed sparse matrix
    void* C,                 // Output matrix (M × N)
    int M, int N, int K,
    SparseFlowDataType dtype,
    void* stream             // CUDA stream (or NULL for default)
);

// Compress dense tensor to 2:4 format
SPARSEFLOW_API SparseFlowStatus sparseflow_compress_2_4(
    SparseFlowContext ctx,
    const void* dense,       // Input dense tensor
    void* compressed,        // Output compressed tensor (50% size)
    void* metadata,          // Output metadata
    int M, int N,
    SparseFlowDataType dtype
);

// Validate 2:4 sparsity pattern
SPARSEFLOW_API SparseFlowStatus sparseflow_validate_2_4(
    const void* tensor,
    int M, int N,
    SparseFlowDataType dtype,
    int* is_valid            // Output: 1 if valid, 0 if not
);

//===----------------------------------------------------------------------===//
// Benchmarking utilities
//===----------------------------------------------------------------------===//

typedef struct {
    double elapsed_ms;       // Elapsed time in milliseconds
    double tflops_effective; // Effective TFLOPS
    double tflops_real;      // Real TFLOPS (accounting for sparsity)
    double bandwidth_gb_s;   // Memory bandwidth (GB/s)
} SparseFlowBenchmarkResult;

// Benchmark kernel performance
SPARSEFLOW_API SparseFlowStatus sparseflow_benchmark_kernel(
    SparseFlowKernel kernel,
    int M, int N, int K,
    int num_iterations,
    SparseFlowBenchmarkResult* result
);

#ifdef __cplusplus
}
#endif

#endif // SPARSEFLOW_C_API_H
