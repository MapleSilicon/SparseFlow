#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__
void sparse_tc_m16n8k32(
    const half* __restrict__ Acomp,   // [M, K/2] row-major
    const uint16_t* __restrict__ E,   // [K/4, M] flattened → E[g*M + row]
    const half* __restrict__ B,       // [K, N] row-major
    float* __restrict__ C,            // [M, N]
    int M, int N, int K
) {
    int lane = threadIdx.x & 31;
    int tile_row = blockIdx.y * 16;
    int tile_col = blockIdx.x * 8;
    
    if (tile_row >= M || tile_col >= N) return;
    
    float d0=0.f, d1=0.f, d2=0.f, d3=0.f;
    float d4=0.f, d5=0.f, d6=0.f, d7=0.f;
    
    for (int k0 = 0; k0 < K; k0 += 32) {
        // Load A tile (16×32 → 8 u32 regs)
        uint32_t a0, a1, a2, a3, a4, a5, a6, a7;
        int a_row = tile_row + (lane >> 2);
        int a_col = k0 / 2 + (lane & 3) * 2;
        const uint32_t* Ap = reinterpret_cast<const uint32_t*>(
            Acomp + a_row * (K/2) + a_col
        );
        a0=Ap[0]; a1=Ap[1]; a2=Ap[2]; a3=Ap[3];
        a4=Ap[4]; a5=Ap[5]; a6=Ap[6]; a7=Ap[7];
        
        // Load B tile (32×8 sparse → 4 u32 regs)
        uint32_t b0, b1, b2, b3;
        int b_row = k0 + (lane >> 3) * 4;
        int b_col = tile_col + (lane & 7);
        const uint32_t* Bp = reinterpret_cast<const uint32_t*>(
            B + b_row * N + b_col
        );
        b0=Bp[0]; b1=Bp[1]; b2=Bp[2]; b3=Bp[3];
        
        // Load metadata
        int meta_g = (k0 / 4) * M + (tile_row + (lane >> 2));
        uint32_t meta = reinterpret_cast<const uint32_t*>(E)[meta_g];
        
        // Sparse Tensor Core MMA
        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, "
            "{%8,%9,%10,%11,%12,%13,%14,%15}, "
            "{%16,%17,%18,%19}, "
            "%20, "
            "{%0,%1,%2,%3,%4,%5,%6,%7};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3),
              "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(a4), "r"(a5), "r"(a6), "r"(a7),
              "r"(b0), "r"(b1), "r"(b2), "r"(b3),
              "r"(meta)
        );
    }
    
    // Store results
    int out_row = tile_row + (lane >> 2);
    int out_col = tile_col + (lane & 3) * 2;
    if (out_row < M && out_col + 1 < N) {
        float* Cp = C + out_row * N + out_col;
        Cp[0] = d0;
        Cp[1] = d1;
    }
}

extern "C" void launch_sparse_tc(
    const half* A, const uint16_t* E, const half* B, float* C,
    int M, int N, int K
) {
    dim3 block(32);
    dim3 grid((N+7)/8, (M+15)/16);
    sparse_tc_m16n8k32<<<grid, block>>>(A, E, B, C, M, N, K);
}
