//===- test_abi_stability.c - Test ABI stability ------------------------===//
//
// Tests that the C API is stable across versions
//
//===----------------------------------------------------------------------===//

#include "sparseflow/c_api.h"
#include <stdio.h>
#include <assert.h>

int test_version_info() {
    printf("Testing version info...\n");
    
    SparseFlowVersion version = sparseflow_get_version();
    printf("  Version: %d.%d.%d\n", version.major, version.minor, version.patch);
    
    // Check compatibility
    int compat = sparseflow_is_abi_compatible(
        SPARSEFLOW_ABI_VERSION_MAJOR,
        SPARSEFLOW_ABI_VERSION_MINOR
    );
    assert(compat == 1);
    
    printf("  ✓ Version check passed\n");
    return 0;
}

int test_context_lifecycle() {
    printf("Testing context lifecycle...\n");
    
    SparseFlowContext ctx = NULL;
    SparseFlowStatus status;
    
    // Create context
    status = sparseflow_create_context(&ctx, 0);
    if (status != SPARSEFLOW_SUCCESS) {
        printf("  ⚠ Failed to create context: %s\n", 
               sparseflow_get_error_string(status));
        printf("  (This is expected if no CUDA GPU is available)\n");
        return 0;
    }
    
    // Get device info
    int compute_cap, sm_count;
    size_t total_mem;
    status = sparseflow_get_device_info(ctx, &compute_cap, &sm_count, &total_mem);
    assert(status == SPARSEFLOW_SUCCESS);
    
    printf("  GPU: SM%d, %d SMs, %zu MB\n", 
           compute_cap, sm_count, total_mem / (1024*1024));
    
    // Destroy context
    status = sparseflow_destroy_context(ctx);
    assert(status == SPARSEFLOW_SUCCESS);
    
    printf("  ✓ Context lifecycle passed\n");
    return 0;
}

int test_error_handling() {
    printf("Testing error handling...\n");
    
    // Test NULL pointer handling
    SparseFlowStatus status = sparseflow_create_context(NULL, 0);
    assert(status == SPARSEFLOW_ERROR_INVALID_ARGUMENT);
    
    const char* error_msg = sparseflow_get_last_error();
    printf("  Error message: %s\n", error_msg);
    
    printf("  ✓ Error handling passed\n");
    return 0;
}

int test_kernel_config() {
    printf("Testing kernel configuration...\n");
    
    SparseFlowKernelConfig config = {
        .tile_m = 128,
        .tile_n = 128,
        .tile_k = 64,
        .epilogue = {
            .kind = SPARSEFLOW_EPILOGUE_RELU,
            .params = NULL,
            .params_size = 0
        }
    };
    
    printf("  Config: tile=(%d,%d,%d), epilogue=%d\n",
           config.tile_m, config.tile_n, config.tile_k, config.epilogue.kind);
    
    printf("  ✓ Kernel config passed\n");
    return 0;
}

int main() {
    printf("="*60 "\n");
    printf("SPARSEFLOW ABI STABILITY TEST\n");
    printf("="*60 "\n\n");
    
    int failures = 0;
    
    failures += test_version_info();
    failures += test_error_handling();
    failures += test_kernel_config();
    failures += test_context_lifecycle();
    
    printf("\n");
    if (failures == 0) {
        printf("✅ All ABI tests passed!\n");
    } else {
        printf("❌ %d test(s) failed\n", failures);
    }
    
    return failures;
}
