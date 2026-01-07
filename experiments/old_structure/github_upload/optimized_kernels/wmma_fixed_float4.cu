#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>

using namespace nvcuda;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// FIXED WMMA KERNEL - Vectorized float4 loading + size_t indexing
__global__ void wmma_kernel_fixed_float4(
    const half* A, const half* B, float* C, 
    const float* bias, int M, int N, int K) {
    
    // Shared Memory with simple padding (avoid complex indexing)
    __shared__ half pAs[32][16];  
    __shared__ half pBs[16][32];  
    
    // WMMA Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Warp identification  
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    // Each warp owns a specific 16x16 tile
    int warpRow = (warpId / 2) * 16;  // 0 or 16
    int warpCol = (warpId % 2) * 16;  // 0 or 16
    
    // Block position - USE size_t to prevent overflow
    size_t blockRow = (size_t)blockIdx.y * 32;
    size_t blockCol = (size_t)blockIdx.x * 32;
    
    // Main K-loop
    for (size_t k = 0; k < K; k += 16) {
        
        // ===== VECTORIZED LOADING OF A TILE (32x16) =====
        // Use float4 for 128-bit coalesced loads (8 halves = 1 float4)
        if (threadIdx.x < 64) {  // Only 64 threads needed for 32x16 tile
            int load_row = threadIdx.x / 2;          // 0..31
            int load_col_base = (threadIdx.x % 2) * 8; // 0 or 8
            
            size_t global_row = blockRow + load_row;
            size_t global_col_base = k + load_col_base;
            
            // Check bounds for vectorized load
            if (global_row < M && (global_col_base + 8) <= K) {
                // VECTORIZED LOAD: 8 halves at once for perfect coalescing
                float4 vec_data = *reinterpret_cast<const float4*>(
                    &A[global_row * K + global_col_base]);
                *reinterpret_cast<float4*>(&pAs[load_row][load_col_base]) = vec_data;
            } else {
                // Scalar loads with bounds checking
                for (int offset = 0; offset < 8; offset++) {
                    size_t global_col = global_col_base + offset;
                    if (global_row < M && global_col < K) {
                        pAs[load_row][load_col_base + offset] = A[global_row * K + global_col];
                    } else {
                        pAs[load_row][load_col_base + offset] = __float2half(0.0f);
                    }
                }
            }
        }
        
        // ===== VECTORIZED LOADING OF B TILE (16x32) =====
        if (threadIdx.x < 64) {  // 64 threads for 16x32 tile
            int load_b_row = threadIdx.x / 4;           // 0..15  
            int load_b_col_base = (threadIdx.x % 4) * 8; // 0, 8, 16, 24
            
            size_t global_b_row = k + load_b_row;
            size_t global_b_col_base = blockCol + load_b_col_base;
            
            if (global_b_row < K && (global_b_col_base + 8) <= N) {
                // VECTORIZED LOAD: 8 halves at once
                float4 vec_b = *reinterpret_cast<const float4*>(
                    &B[global_b_row * N + global_b_col_base]);
                *reinterpret_cast<float4*>(&pBs[load_b_row][load_b_col_base]) = vec_b;
            } else {
                // Scalar loads with bounds checking
                for (int offset = 0; offset < 8; offset++) {
                    size_t global_b_col = global_b_col_base + offset;
                    if (global_b_row < K && global_b_col < N) {
                        pBs[load_b_row][load_b_col_base + offset] = 
                            B[global_b_row * N + global_b_col];
                    } else {
                        pBs[load_b_row][load_b_col_base + offset] = __float2half(0.0f);
                    }
                }
            }
        }
        
        __syncthreads();  // Wait for shared memory loads
        
        // ===== WARP-SPECIFIC WMMA COMPUTATION =====
        // Load fragments - use simple stride (no complex padding)
        wmma::load_matrix_sync(a_frag, &pAs[warpRow][0], 16);
        wmma::load_matrix_sync(b_frag, &pBs[0][warpCol], 32);
        
        // Tensor Core computation
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        __syncthreads();  // Wait before next iteration
    }
    
    // ===== STORE RESULTS =====
    size_t outRow = blockRow + warpRow;
    size_t outCol = blockCol + warpCol;
    
    if (outRow < M && outCol < N) {
        // Shared memory for bias+ReLU
        __shared__ float results[4][256];  
        
        // Store WMMA result  
        wmma::store_matrix_sync(&results[warpId][0], acc_frag, 16, wmma::mem_row_major);
        
        __syncthreads();
        
        // Apply bias+ReLU (8 elements per thread)
        for (int i = 0; i < 8; i++) {
            int elem_idx = laneId + i * 32;  // Stride by 32 to cover all elements
            if (elem_idx < 256) {
                int elem_row = elem_idx / 16;
                int elem_col = elem_idx % 16;
                
                size_t global_col_bias = outCol + elem_col;
                float bias_val = (global_col_bias < N) ? bias[global_col_bias] : 0.0f;
                
                float val = results[warpId][elem_idx];
                float biased = val + bias_val;
                results[warpId][elem_idx] = fmaxf(biased, 0.0f);
            }
        }
        
        __syncthreads();
        
        // Write to global memory with size_t indexing
        for (int i = 0; i < 8; i++) {
            int elem_idx = laneId + i * 32;
            if (elem_idx < 256) {
                int elem_row = elem_idx / 16;
                int elem_col = elem_idx % 16;
                
                size_t final_row = outRow + elem_row;
                size_t final_col = outCol + elem_col;
                
                if (final_row < M && final_col < N) {
                    C[final_row * N + final_col] = results[warpId][elem_idx];
                }
            }
        }
    }
}

// Utility functions
void convert_fp32_to_fp16(const float* src, half* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = __float2half(src[i]);
    }
}

void benchmark_fixed_wmma(int M, int N, int K) {
    std::cout << "\n=== FIXED WMMA (float4): " << M << "x" << N << "x" << K << " ===\n";
    
    // Allocate memory
    std::vector<float> h_A(M * K), h_B(K * N), h_bias(N, 0.01f);  // Smaller bias
    std::vector<half> h_A_fp16(M * K), h_B_fp16(K * N);
    std::vector<float> h_C(M * N), h_ref(M * N);
    
    // Initialize with smaller values for better FP16 precision
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    
    for (size_t i = 0; i < M * K; i++) h_A[i] = dist(gen);
    for (size_t i = 0; i < K * N; i++) h_B[i] = dist(gen);
    
    convert_fp32_to_fp16(h_A.data(), h_A_fp16.data(), M * K);
    convert_fp32_to_fp16(h_B.data(), h_B_fp16.data(), K * N);
    
    // Device memory
    half *d_A, *d_B;
    float *d_C, *d_bias;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A_fp16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_fp16.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 block(128);  // 4 warps
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    
    std::cout << "Grid: (" << grid.x << ", " << grid.y << "), Block: " << block.x << "\n";
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        wmma_kernel_fixed_float4<<<grid, block>>>(d_A, d_B, d_C, d_bias, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        wmma_kernel_fixed_float4<<<grid, block>>>(d_A, d_B, d_C, d_bias, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double>(end - start).count() / 10.0;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = flops / (duration * 1e12);
    
    std::cout << "Time: " << duration * 1000 << " ms\n";
    std::cout << "TFLOPS: " << tflops << "\n";
    
    // Get results and verify
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // CPU reference  
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_ref[i * N + j] = fmaxf(sum + h_bias[j], 0.0f);
        }
    }
    
    // Verification
    float max_error = 0;
    float max_rel_error = 0;
    int errors = 0;
    
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C[i] - h_ref[i]);
        max_error = fmax(max_error, error);
        
        if (fabs(h_ref[i]) > 1e-6f) {
            float rel_error = error / fabs(h_ref[i]);
            max_rel_error = fmax(max_rel_error, rel_error);
            if (rel_error > 0.05f) errors++;  // 5% threshold
        }
    }
    
    std::cout << "Max error: " << max_error << "\n";
    std::cout << "Max rel error: " << max_rel_error * 100 << "%\n";
    std::cout << "High errors: " << errors << " / " << (M * N) << "\n";
    
    if (max_error < 0.5f && errors < (M * N) * 0.01) {
        std::cout << "✅ PASSED verification!\n";
        std::cout << "🚀 TARGET: 40+ TFLOPS " << (tflops >= 40.0 ? "ACHIEVED!" : "needs optimization") << "\n";
    } else {
        std::cout << "❌ FAILED verification\n";
    }
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_bias);
}

int main() {
    std::cout << "=== WMMA FLOAT4 VECTORIZED BENCHMARK ===\n";
    std::cout << "Goal: Fix indexing + achieve 40+ TFLOPS\n\n";
    
    int sizes[] = {256, 512, 1024, 2048};
    for (int size : sizes) {
        size_t mem_req = (size_t)size * size * 10;
        if (mem_req > 10e9) {  // 10GB limit
            std::cout << "\nSkipping " << size << " (memory limit)\n";
            continue;
        }
        benchmark_fixed_wmma(size, size, size);
    }
    
    return 0;
}
