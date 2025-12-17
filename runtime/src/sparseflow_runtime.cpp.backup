#include "sparseflow_runtime.h"
#include <cstring>
#include <cstdlib>
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

// Template kernel implementation (NOT in extern "C")
template<int N_PATTERN, int M_PATTERN>
static void sparse_matmul_nm_impl(
    float* C, const float* A, const float* B,
    int M, int K, int N_dim,
    const bool* rowmask_A = nullptr,
    const bool* colmask_B = nullptr) {
    
    memset(C, 0, sizeof(float) * M * N_dim);
    
#pragma omp parallel for if(M > 16)
    for (int i = 0; i < M; i++) {
        if (rowmask_A && !rowmask_A[i]) continue;
        
        if (M_PATTERN > 0) {
            int pos_in_block = i % M_PATTERN;
            if (pos_in_block >= N_PATTERN) continue;
        }
        
        for (int j = 0; j < N_dim; j++) {
            if (colmask_B && !colmask_B[j]) continue;
            
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N_dim + j];
            }
            C[i*N_dim + j] = sum;
        }
    }
}

// Now the extern "C" block with exported functions
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

// All 5 N:M kernels
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

// Validation functions
bool validate_nm_pattern(
    const float* tensor,
    int rows, int cols,
    int N, int M,
    float zero_threshold) {
    
    if (N > M || M <= 0 || N <= 0) return false;
    
    for (int i = 0; i < rows; i += M) {
        int block_size = (i + M <= rows) ? M : (rows - i);
        for (int j = 0; j < cols; j++) {
            int nonzeros = 0;
            for (int bi = 0; bi < block_size; bi++) {
                float val = tensor[(i + bi) * cols + j];
                if (val > zero_threshold || val < -zero_threshold) {
                    nonzeros++;
                }
            }
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
    float zero_threshold) {
    
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
    
    for (int i = 0; i < rows; i += M) {
        int block_size = (i + M <= rows) ? M : (rows - i);
        if (block_size != M) continue;
        
        for (int j = 0; j < cols; j++) {
            result.total_blocks++;
            int nonzeros = 0;
            
            for (int bi = 0; bi < M; bi++) {
                float val = tensor[(i + bi) * cols + j];
                if (val > zero_threshold || val < -zero_threshold) {
                    nonzeros++;
                }
            }
            
            total_nonzeros += nonzeros;
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

} // extern "C"
