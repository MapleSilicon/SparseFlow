//===- context.cpp - Context implementation -----------------------------===//

#include "sparseflow/c_api.h"
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <mutex>
#include <memory>

namespace sparseflow {
namespace internal {

// Thread-local error storage
thread_local std::string last_error;

void set_last_error(const std::string& error) {
    last_error = error;
}

// Context implementation (hidden from users)
struct ContextImpl {
    int device_id;
    cudaDeviceProp device_props;
    std::map<std::string, void*> kernel_cache;
    std::mutex mutex;
    
    ContextImpl(int dev_id) : device_id(dev_id) {
        cudaGetDeviceProperties(&device_props, dev_id);
    }
    
    ~ContextImpl() {
        // Cleanup cached kernels
        for (auto& kv : kernel_cache) {
            // Free kernel resources
        }
    }
    
    int get_compute_capability() const {
        return device_props.major * 10 + device_props.minor;
    }
    
    int get_sm_count() const {
        return device_props.multiProcessorCount;
    }
    
    size_t get_total_memory() const {
        return device_props.totalGlobalMem;
    }
};

// Kernel implementation (hidden from users)
struct KernelImpl {
    std::shared_ptr<ContextImpl> ctx;
    SparseFlowKernelConfig config;
    void* cuda_function;
    dim3 grid;
    dim3 block;
    
    KernelImpl(std::shared_ptr<ContextImpl> c, const SparseFlowKernelConfig& cfg)
        : ctx(c), config(cfg), cuda_function(nullptr) {}
    
    ~KernelImpl() {
        // Cleanup CUDA resources
    }
};

} // namespace internal
} // namespace sparseflow

using namespace sparseflow::internal;

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

extern "C" {

const char* sparseflow_get_error_string(SparseFlowStatus status) {
    switch(status) {
        case SPARSEFLOW_SUCCESS: return "Success";
        case SPARSEFLOW_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case SPARSEFLOW_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case SPARSEFLOW_ERROR_CUDA_ERROR: return "CUDA error";
        case SPARSEFLOW_ERROR_NOT_SUPPORTED: return "Not supported";
        case SPARSEFLOW_ERROR_INVALID_HANDLE: return "Invalid handle";
        case SPARSEFLOW_ERROR_COMPILATION_FAILED: return "Compilation failed";
        case SPARSEFLOW_ERROR_ABI_MISMATCH: return "ABI version mismatch";
        default: return "Unknown error";
    }
}

const char* sparseflow_get_last_error() {
    return last_error.c_str();
}

//===----------------------------------------------------------------------===//
// Version info
//===----------------------------------------------------------------------===//

SparseFlowVersion sparseflow_get_version() {
    return {
        SPARSEFLOW_ABI_VERSION_MAJOR,
        SPARSEFLOW_ABI_VERSION_MINOR,
        SPARSEFLOW_ABI_VERSION_PATCH
    };
}

int sparseflow_is_abi_compatible(int major, int minor) {
    // Compatible if major version matches
    return (major == SPARSEFLOW_ABI_VERSION_MAJOR) ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

SparseFlowStatus sparseflow_create_context(
    SparseFlowContext* ctx,
    int device_id
) {
    if (!ctx) {
        set_last_error("Context pointer is NULL");
        return SPARSEFLOW_ERROR_INVALID_ARGUMENT;
    }
    
    // Check device exists
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        set_last_error("Failed to get device count");
        return SPARSEFLOW_ERROR_CUDA_ERROR;
    }
    
    if (device_id < 0 || device_id >= device_count) {
        set_last_error("Invalid device ID");
        return SPARSEFLOW_ERROR_INVALID_ARGUMENT;
    }
    
    // Create context
    try {
        auto impl = std::make_shared<ContextImpl>(device_id);
        *ctx = reinterpret_cast<SparseFlowContext>(
            new std::shared_ptr<ContextImpl>(impl)
        );
        return SPARSEFLOW_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return SPARSEFLOW_ERROR_OUT_OF_MEMORY;
    }
}

SparseFlowStatus sparseflow_destroy_context(SparseFlowContext ctx) {
    if (!ctx) {
        return SPARSEFLOW_ERROR_INVALID_HANDLE;
    }
    
    try {
        auto* impl_ptr = reinterpret_cast<std::shared_ptr<ContextImpl>*>(ctx);
        delete impl_ptr;
        return SPARSEFLOW_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return SPARSEFLOW_ERROR_INVALID_HANDLE;
    }
}

SparseFlowStatus sparseflow_get_device_info(
    SparseFlowContext ctx,
    int* compute_capability,
    int* sm_count,
    size_t* total_memory
) {
    if (!ctx) {
        return SPARSEFLOW_ERROR_INVALID_HANDLE;
    }
    
    try {
        auto& impl = **reinterpret_cast<std::shared_ptr<ContextImpl>*>(ctx);
        
        if (compute_capability) {
            *compute_capability = impl.get_compute_capability();
        }
        if (sm_count) {
            *sm_count = impl.get_sm_count();
        }
        if (total_memory) {
            *total_memory = impl.get_total_memory();
        }
        
        return SPARSEFLOW_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return SPARSEFLOW_ERROR_INVALID_HANDLE;
    }
}

//===----------------------------------------------------------------------===//
// Kernel compilation
//===----------------------------------------------------------------------===//

SparseFlowStatus sparseflow_compile_kernel(
    SparseFlowContext ctx,
    SparseFlowKernel* kernel,
    const SparseFlowKernelConfig* config
) {
    if (!ctx || !kernel || !config) {
        set_last_error("NULL pointer argument");
        return SPARSEFLOW_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto ctx_impl = *reinterpret_cast<std::shared_ptr<ContextImpl>*>(ctx);
        auto kernel_impl = std::make_shared<KernelImpl>(ctx_impl, *config);
        
        // TODO: Actually compile kernel based on config
        // For now, just store the config
        
        *kernel = reinterpret_cast<SparseFlowKernel>(
            new std::shared_ptr<KernelImpl>(kernel_impl)
        );
        
        return SPARSEFLOW_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return SPARSEFLOW_ERROR_COMPILATION_FAILED;
    }
}

SparseFlowStatus sparseflow_destroy_kernel(SparseFlowKernel kernel) {
    if (!kernel) {
        return SPARSEFLOW_ERROR_INVALID_HANDLE;
    }
    
    try {
        auto* impl_ptr = reinterpret_cast<std::shared_ptr<KernelImpl>*>(kernel);
        delete impl_ptr;
        return SPARSEFLOW_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return SPARSEFLOW_ERROR_INVALID_HANDLE;
    }
}

//===----------------------------------------------------------------------===//
// Tensor operations (stubs for now)
//===----------------------------------------------------------------------===//

SparseFlowStatus sparseflow_sparse_gemm(
    SparseFlowKernel kernel,
    const void* A,
    const void* Bc,
    void* C,
    int M, int N, int K,
    SparseFlowDataType dtype,
    void* stream
) {
    if (!kernel || !A || !Bc || !C) {
        set_last_error("NULL pointer in sparse_gemm");
        return SPARSEFLOW_ERROR_INVALID_ARGUMENT;
    }
    
    // TODO: Call actual kernel
    set_last_error("sparse_gemm not yet implemented");
    return SPARSEFLOW_ERROR_NOT_SUPPORTED;
}

SparseFlowStatus sparseflow_compress_2_4(
    SparseFlowContext ctx,
    const void* dense,
    void* compressed,
    void* metadata,
    int M, int N,
    SparseFlowDataType dtype
) {
    // TODO: Implement compression
    set_last_error("compress_2_4 not yet implemented");
    return SPARSEFLOW_ERROR_NOT_SUPPORTED;
}

SparseFlowStatus sparseflow_validate_2_4(
    const void* tensor,
    int M, int N,
    SparseFlowDataType dtype,
    int* is_valid
) {
    // TODO: Implement validation
    set_last_error("validate_2_4 not yet implemented");
    return SPARSEFLOW_ERROR_NOT_SUPPORTED;
}

SparseFlowStatus sparseflow_benchmark_kernel(
    SparseFlowKernel kernel,
    int M, int N, int K,
    int num_iterations,
    SparseFlowBenchmarkResult* result
) {
    // TODO: Implement benchmarking
    set_last_error("benchmark_kernel not yet implemented");
    return SPARSEFLOW_ERROR_NOT_SUPPORTED;
}

} // extern "C"
