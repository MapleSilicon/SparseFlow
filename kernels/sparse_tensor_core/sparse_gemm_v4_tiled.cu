#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp16.h>

// Tile sizes
#define TILE_M 128
#define TILE_N 128
#define TILE_K 32

extern "C" __global__ void sparse_gemm_v4_tiled(
    const half* __restrict__ A,      // [M, K/2] compressed
    const uint16_t* __restrict__ E,  // [K/4, M] metadata
    const half* __restrict__ B,      // [K, N] dense
    float* __restrict__ C,           // [M, N] output
    int M, int N, int K) 
{
    // Shared memory
    __shared__ half shm_A[TILE_M * (TILE_K / 2)];  // 128 x 16 = 2048 halves = 4KB
    __shared__ half shm_B[TILE_K * TILE_N];        // 32 x 128 = 4096 halves = 8KB
    
    // Block coordinates
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    int tid = threadIdx.x;
    
    // Thread coordinates within block
    int local_row = tid / TILE_N;  // 0-127
    int local_col = tid % TILE_N;  // 0-127
    
    // Global output coordinates
    int global_row = block_row + local_row;
    int global_col = block_col + local_col;
    
    float accum = 0.0f;
    
    // Loop over K in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        
        // === COOPERATIVE LOAD A TILE ===
        // Load compressed A: [TILE_M, TILE_K/2]
        int num_a_elements = TILE_M * (TILE_K / 2);  // 128 * 16 = 2048
        for (int i = tid; i < num_a_elements; i += blockDim.x) {
            int a_row = i / (TILE_K / 2);
            int a_col = i % (TILE_K / 2);
            int global_a_row = block_row + a_row;
            int global_a_col = (k_tile / 2) + a_col;
            
            if (global_a_row < M && global_a_col < K/2) {
                shm_A[i] = A[global_a_row * (K/2) + global_a_col];
            } else {
                shm_A[i] = __float2half(0.0f);
            }
        }
        
        // === COOPERATIVE LOAD B TILE ===
        // Load dense B: [TILE_K, TILE_N]
        int num_b_elements = TILE_K * TILE_N;  // 32 * 128 = 4096
        for (int i = tid; i < num_b_elements; i += blockDim.x) {
            int b_row = i / TILE_N;
            int b_col = i % TILE_N;
            int global_b_row = k_tile + b_row;
            int global_b_col = block_col + b_col;
            
            if (global_b_row < K && global_b_col < N) {
                shm_B[i] = B[global_b_row * N + global_b_col];
            } else {
                shm_B[i] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // === COMPUTE FROM SHARED MEMORY ===
        // Each thread computes ONE output element using shared memory
        if (global_row < M && global_col < N) {
            // Loop over the K dimension within this tile
            for (int k = 0; k < TILE_K; k += 4) {
                int g = (k_tile + k) / 4;  // Global group index
                
                // Get metadata
                uint16_t code = E[g * M + global_row];
                
                // Decode
                int i0, i1;
                switch(code) {
                    case 0: i0=0; i1=1; break;
                    case 1: i0=0; i1=2; break;
                    case 2: i0=0; i1=3; break;
                    case 3: i0=1; i1=2; break;
                    case 4: i0=1; i1=3; break;
                    case 5: i0=2; i1=3; break;
                    default: i0=0; i1=2; break;
                }
                
                // Local indices within shared memory
                int local_a_offset = local_row * (TILE_K / 2) + (k / 2);
                int local_b0_offset = (k + i0) * TILE_N + local_col;
                int local_b1_offset = (k + i1) * TILE_N + local_col;
                
                half a0 = shm_A[local_a_offset + 0];
                half a1 = shm_A[local_a_offset + 1];
                half b0 = shm_B[local_b0_offset];
                half b1 = shm_B[local_b1_offset];
                
                accum += __half2float(a0) * __half2float(b0);
                accum += __half2float(a1) * __half2float(b1);
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = accum;
    }
}

extern "C" void launch_sparse_v4_tiled(
    const half* A, const uint16_t* E, const half* B, float* C,
    int M, int N, int K)
{
    // Each block handles 128x128 output
    // Need enough threads to cover the block
    dim3 block(256);  // 256 threads per block
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    sparse_gemm_v4_tiled<<<grid, block>>>(A, E, B, C, M, N, K);
}
