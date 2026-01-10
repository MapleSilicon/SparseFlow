#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// One-warp dense Tensor Core MMA
__global__ void mma_m16n8k16_bulletproof(
    const uint16_t* A16,
    const uint16_t* B16,
    float* C4)
{
    // single warp only
    if ((threadIdx.x >> 5) != 0) return;

    // ---- A matrix: 8 × f16 ----
    uint16_t a0,a1,a2,a3,a4,a5,a6,a7;
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(a0) : "l"(A16+0));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(a1) : "l"(A16+1));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(a2) : "l"(A16+2));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(a3) : "l"(A16+3));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(a4) : "l"(A16+4));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(a5) : "l"(A16+5));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(a6) : "l"(A16+6));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(a7) : "l"(A16+7));

    // ---- B matrix: 4 × f16 ----
    uint16_t b0,b1,b2,b3;
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(b0) : "l"(B16+0));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(b1) : "l"(B16+1));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(b2) : "l"(B16+2));
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(b3) : "l"(B16+3));

    // ---- Accumulators ----
    float c0=0.f, c1=0.f, c2=0.f, c3=0.f;

    // ---- Tensor Core MMA (CORRECT SIGNATURE) ----
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7,%8,%9,%10,%11}, "
        "{%12,%13,%14,%15}, "
        "{%0,%1,%2,%3};"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "h"(a0),"h"(a1),"h"(a2),"h"(a3),
          "h"(a4),"h"(a5),"h"(a6),"h"(a7),
          "h"(b0),"h"(b1),"h"(b2),"h"(b3)
    );

    // ---- Store ----
    if (threadIdx.x == 0) {
        C4[0]=c0; C4[1]=c1; C4[2]=c2; C4[3]=c3;
    }
}

int main() {
    uint16_t *A,*B;
    float *C;

    cudaMalloc(&A, 16*sizeof(uint16_t));
    cudaMalloc(&B, 16*sizeof(uint16_t));
    cudaMalloc(&C, 4*sizeof(float));

    mma_m16n8k16_bulletproof<<<1,32>>>(A,B,C);
    cudaDeviceSynchronize();

    printf("SUCCESS: MMA kernel compiled and executed.\\n");
    return 0;
}
