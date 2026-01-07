#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    // Ampere (8.x) and later support sparse Tensor Cores
    if (prop.major >= 8) {
        printf("✅ Supports 2:4 Sparse Tensor Cores\n");
        printf("Theoretical sparse FP16 peak: ~70 TFLOPS\n");
    } else {
        printf("❌ Does NOT support sparse Tensor Cores\n");
    }
    
    return 0;
}
