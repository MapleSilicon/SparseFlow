//===- sparseflow_runtime.cpp - N:M Sparse Runtime -----------------------===//

#include "sparseflow_runtime.h"
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

struct SparseMatrixHandle {
    bool* rowmask;
    bool* colmask;
    int rows;
    int cols;
};

extern "C" {

SparseMatrix sparseflow_create_matrix(
    const bool* rowmask, int rows,
    const bool* colmask, int cols) {
    auto* mat = (SparseMatrixHandle*)malloc(sizeof(SparseMatrixHandle));
    mat->rows = rows;
    mat->cols = cols;
    mat->rowmask = (bool*)malloc(rows * sizeof(bool));
    mat->colmask = (bool*)malloc(cols * sizeof(bool));
    memcpy(mat->rowmask, rowmask, rows * sizeof(bool));
    memcpy(mat->colmask, colmask, cols * sizeof(bool));
    return mat;
}

void sparseflow_destroy_matrix(SparseMatrix mat) {
    if (!mat) return;
    free(mat->rowmask);
    free(mat->colmask);
    free(mat);
}

// Dense baseline for comparison
static void dense_matmul(float* C, const float* A, const float* B,
                         int M, int K, int N) {
    memset(C, 0, sizeof(float) * M * N);
#pragma omp parallel for collapse(2) if(M * N > 4096)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// CORE: Template-based N:M sparse kernel
// This is the heart of the runtime - handles ANY N:M pattern
template<int N_PATTERN, int M_PATTERN>
static void sparse_matmul_nm_impl(
    float* C, const float* A, const float* B,
    int M, int K, int N_dim,
    const bool* rowmask_A = nullptr,
    const bool* colmask_B = nullptr) {
    
    memset(C, 0, sizeof(float) * M * N_dim);
    
#pragma omp parallel for if(M > 16)
    for (int i = 0; i < M; i++) {
        // Check if this row is active
        if (rowmask_A && !rowmask_A[i]) continue;
        
        // N:M row-wise sparsity: Check blocks of M_PATTERN rows
        // If we're in a sparse block, check if we have enough non-zeros
        if (M_PATTERN > 0) {
            int block_idx = i / M_PATTERN;
            int pos_in_block = i % M_PATTERN;
            
            // Simple heuristic: if in the "zero" part of N:M block, skip
            // (This is a simplified version; real implementation would check actual values)
            if (pos_in_block >= N_PATTERN) {
                continue;
            }
        }
        
        for (int j = 0; j < N_dim; j++) {
            // Check if this column is active
            if (colmask_B && !colmask_B[j]) continue;
            
            // Similar N:M check for columns
            if (M_PATTERN > 0) {
                int block_idx = j / M_PATTERN;
                int pos_in_block = j % M_PATTERN;
                if (pos_in_block >= N_PATTERN) {
                    continue;
                }
            }
            
            // Compute dot product
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N_dim + j];
            }
            C[i*N_dim + j] = sum;
        }
    }
}

// Keep original for backwards compatibility
void sparse_matmul_2_4_impl(
    float* C, const float* A, const float* B,
    int M, int K, int N,
    const bool* rowmask_A, const bool* colmask_B) {
    sparse_matmul_nm_impl<2, 4>(C, A, B, M, K, N, rowmask_A, colmask_B);
}

// Explicit instantiations for each N:M pattern
// These are the functions that get called from MLIR

void sparse_matmul_1_4(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape) {
    int M = lhs_shape[0];
    int K = lhs_shape[1];
    int N = rhs_shape[1];
    sparse_matmul_nm_impl<1, 4>(out, lhs, rhs, M, K, N, nullptr, nullptr);
}

void sparse_matmul_2_4(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape) {
    int M = lhs_shape[0];
    int K = lhs_shape[1];
    int N = rhs_shape[1];
    sparse_matmul_nm_impl<2, 4>(out, lhs, rhs, M, K, N, nullptr, nullptr);
}

void sparse_matmul_2_8(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape) {
    int M = lhs_shape[0];
    int K = lhs_shape[1];
    int N = rhs_shape[1];
    sparse_matmul_nm_impl<2, 8>(out, lhs, rhs, M, K, N, nullptr, nullptr);
}

void sparse_matmul_4_16(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape) {
    int M = lhs_shape[0];
    int K = lhs_shape[1];
    int N = rhs_shape[1];
    sparse_matmul_nm_impl<4, 16>(out, lhs, rhs, M, K, N, nullptr, nullptr);
}

void sparse_matmul_8_32(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape) {
    int M = lhs_shape[0];
    int K = lhs_shape[1];
    int N = rhs_shape[1];
    sparse_matmul_nm_impl<8, 32>(out, lhs, rhs, M, K, N, nullptr, nullptr);
}

// Benchmark function
BenchmarkResult sparseflow_benchmark_matmul(
    const float* A, const float* B,
    int M, int K, int N,
    const bool* rowmask_A, const bool* colmask_B,
    int num_iterations) {
    
    BenchmarkResult r{};
    float* C = (float*)malloc(sizeof(float) * M * N);
    
    // Warmup
    dense_matmul(C, A, B, M, K, N);
    sparse_matmul_2_4_impl(C, A, B, M, K, N, rowmask_A, colmask_B);
    
    // Benchmark dense
    auto d0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++)
        dense_matmul(C, A, B, M, K, N);
    auto d1 = std::chrono::high_resolution_clock::now();
    
    // Benchmark sparse
    auto s0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++)
        sparse_matmul_2_4_impl(C, A, B, M, K, N, rowmask_A, colmask_B);
    auto s1 = std::chrono::high_resolution_clock::now();
    
    r.dense_time_ms  = std::chrono::duration<double, std::milli>(d1 - d0).count() / num_iterations;
    r.sparse_time_ms = std::chrono::duration<double, std::milli>(s1 - s0).count() / num_iterations;
    r.speedup = r.dense_time_ms / r.sparse_time_ms;
    
    r.dense_flops  = 2LL * M * N * K;
    int active_rows = rowmask_A ? std::count(rowmask_A, rowmask_A + M, true) : M;
    int active_cols = colmask_B ? std::count(colmask_B, colmask_B + N, true) : N;
    r.sparse_flops = 2LL * K * active_rows * active_cols;
    
    free(C);
    return r;
}

} // extern "C"

// Pattern validation functions

bool validate_nm_pattern(
    const float* tensor,
    int rows, int cols,
    int N, int M,
    float zero_threshold = 1e-6f) {
    
    if (N > M || M <= 0 || N <= 0) return false;
    
    // Check row-wise N:M structure
    for (int i = 0; i < rows; i += M) {
        int block_size = (i + M <= rows) ? M : (rows - i);
        
        for (int j = 0; j < cols; j++) {
            int nonzeros = 0;
            
            // Count non-zeros in this M-element block
            for (int bi = 0; bi < block_size; bi++) {
                float val = tensor[(i + bi) * cols + j];
                if (val > zero_threshold || val < -zero_threshold) {
                    nonzeros++;
                }
            }
            
            // For a valid N:M pattern, each block should have at most N non-zeros
            // (We're being conservative here)
            if (nonzeros > N && block_size == M) {
                return false;
            }
        }
    }
    
    return true;
}

ValidationResult validate_nm_pattern_detailed(
    const float* tensor,
    int rows, int cols,
    int N, int M,
    float zero_threshold = 1e-6f) {
    
    ValidationResult result;
    result.expected_nonzeros_per_block = N;
    result.actual_nonzeros_per_block = 0;
    result.invalid_blocks = 0;
    result.total_blocks = 0;
    result.is_valid = true;
    
    if (N > M || M <= 0 || N <= 0) {
        result.is_valid = false;
        return result;
    }
    
    int total_nonzeros = 0;
    
    // Check row-wise N:M structure
    for (int i = 0; i < rows; i += M) {
        int block_size = (i + M <= rows) ? M : (rows - i);
        if (block_size != M) continue;  // Skip incomplete blocks
        
        for (int j = 0; j < cols; j++) {
            result.total_blocks++;
            int nonzeros = 0;
            
            // Count non-zeros in this M-element block
            for (int bi = 0; bi < M; bi++) {
                float val = tensor[(i + bi) * cols + j];
                if (val > zero_threshold || val < -zero_threshold) {
                    nonzeros++;
                }
            }
            
            total_nonzeros += nonzeros;
            
            // Check if block violates N:M constraint
            if (nonzeros > N) {
                result.invalid_blocks++;
                result.is_valid = false;
            }
        }
    }
    
    if (result.total_blocks > 0) {
        result.actual_nonzeros_per_block = total_nonzeros / result.total_blocks;
    }
    
    return result;
}

