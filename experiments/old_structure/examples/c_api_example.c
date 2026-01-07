//===- c_api_example.c - Using SparseFlow C API -------------------------===//

#include "sparseflow/c_api.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("SparseFlow C API Example\n");
    printf("========================\n\n");
    
    // Check version
    SparseFlowVersion version = sparseflow_get_version();
    printf("Library version: %d.%d.%d\n", 
           version.major, version.minor, version.patch);
    
    // Check ABI compatibility
    if (!sparseflow_is_abi_compatible(1, 0)) {
        fprintf(stderr, "ABI version mismatch!\n");
        return 1;
    }
    printf("ABI: Compatible\n\n");
    
    // Create context
    SparseFlowContext ctx;
    SparseFlowStatus status = sparseflow_create_context(&ctx, 0);
    
    if (status != SPARSEFLOW_SUCCESS) {
        fprintf(stderr, "Failed to create context: %s\n",
                sparseflow_get_error_string(status));
        fprintf(stderr, "Details: %s\n", sparseflow_get_last_error());
        return 1;
    }
    
    // Get GPU info
    int compute_cap, sm_count;
    size_t total_mem;
    sparseflow_get_device_info(ctx, &compute_cap, &sm_count, &total_mem);
    
    printf("GPU Information:\n");
    printf("  Compute capability: SM%d\n", compute_cap);
    printf("  SM count: %d\n", sm_count);
    printf("  Total memory: %.2f GB\n", total_mem / (1024.0*1024.0*1024.0));
    
    // Check 2:4 sparse support
    if (compute_cap >= 80) {
        printf("  ✓ 2:4 sparse supported\n");
    } else {
        printf("  ✗ 2:4 sparse NOT supported (requires SM80+)\n");
    }
    
    printf("\n");
    
    // Configure kernel
    SparseFlowKernelConfig config = {
        .tile_m = 128,
        .tile_n = 128,
        .tile_k = 64,
        .epilogue = {
            .kind = SPARSEFLOW_EPILOGUE_SILU,
            .params = NULL,
            .params_size = 0
        }
    };
    
    SparseFlowKernel kernel;
    status = sparseflow_compile_kernel(ctx, &kernel, &config);
    
    if (status == SPARSEFLOW_SUCCESS) {
        printf("Kernel compiled successfully\n");
        printf("  Configuration: %dx%dx%d tiles\n", 
               config.tile_m, config.tile_n, config.tile_k);
        printf("  Epilogue: SiLU\n");
        
        // Clean up kernel
        sparseflow_destroy_kernel(kernel);
    } else {
        printf("Kernel compilation not yet implemented\n");
    }
    
    // Clean up context
    sparseflow_destroy_context(ctx);
    
    printf("\n✅ Example complete!\n");
    return 0;
}
