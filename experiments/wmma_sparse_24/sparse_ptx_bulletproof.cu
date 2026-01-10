#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__global__ void mma_m16n8k16_packed(
    const uint32_t* A32, // Input A treated as bits (packed u32)
    const uint32_t* B32, // Input B treated as bits (packed u32)
    float* C4)
{
    // Single warp only
    if ((threadIdx.x >> 5) != 0) return;

    // ---- A: 4 x u32 (contains 8 FP16 values packed) ----
    uint32_t a0, a1, a2, a3;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(a0) : "l"(A32 + 0));
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(a1) : "l"(A32 + 1));
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(a2) : "l"(A32 + 2));
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(a3) : "l"(A32 + 3));

    // ---- B: 2 x u32 (contains 4 FP16 values packed) ----
    uint32_t b0, b1;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(b0) : "l"(B32 + 0));
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(b1) : "l"(B32 + 1));

    // ---- Accumulators ----
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    // ---- CORRECT MMA (Packed Operands) ----
    // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // expects A as 4 regs, B as 2 regs.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "       // D (4 x f32)
        "{%4,%5,%6,%7}, "       // A (4 x u32 packed with f16) -- FIXED COUNT
        "{%8,%9}, "             // B (2 x u32 packed with f16) -- FIXED COUNT
        "{%0,%1,%2,%3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );

    // Store (Thread 0 only)
    if (threadIdx.x == 0) {
        C4[0] = c0;
        C4[1] = c1;
        C4[2] = c2;
        C4[3] = c3;
    }
}

int main() {
    // Allocate as u32 (32-bit words) to match the packed loading
    uint32_t *A, *B;
    float *C;
    
    // A needs 4 u32s, B needs 2 u32s for this minimal warp demo
    cudaMalloc(&A, 4 * sizeof(uint32_t));
    cudaMalloc(&B, 2 * sizeof(uint32_t));
    cudaMalloc(&C, 4 * sizeof(float));

    printf("Launching kernel (Packed Registers)...\n");
    mma_m16n8k16_packed<<<1, 32>>>(A, B, C);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        printf("✅ SUCCESS: Kernel compiled and executed.\n");
    } else {
        printf("❌ FAILURE: %s\n", cudaGetErrorString(err));
    }
    
    return 0;
}
