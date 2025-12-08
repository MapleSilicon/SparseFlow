//===- sparseflow_runtime.cpp - SparseFlow Runtime -----------------------===//

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

static void dense_matmul(float* C, const float* A, const float* B,
                         int M, int K, int N) {
    memset(C, 0, sizeof(float) * M * N);
#pragma omp parallel for collapse(2) if(M * N > 4096)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i*K + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

void sparse_matmul_2_4_impl(
    float* C, const float* A, const float* B,
    int M, int K, int N,
    const bool* rowmask_A, const bool* colmask_B) {
    memset(C, 0, sizeof(float) * M * N);
#pragma omp parallel for if(M > 16)
    for (int i = 0; i < M; i++) {
        if (rowmask_A && !rowmask_A[i]) continue;
        for (int j = 0; j < N; j++) {
            if (colmask_B && !colmask_B[j]) continue;
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i*K + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
    }
}

void sparse_matmul_2_4(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape) {
    int M = lhs_shape[0];
    int K = lhs_shape[1];
    int N = rhs_shape[1];
    sparse_matmul_2_4_impl(out, lhs, rhs, M, K, N, nullptr, nullptr);
}

BenchmarkResult sparseflow_benchmark_matmul(
    const float* A, const float* B,
    int M, int K, int N,
    const bool* rowmask_A, const bool* colmask_B,
    int num_iterations) {
    BenchmarkResult r{};
    float* C = (float*)malloc(sizeof(float) * M * N);
    
    dense_matmul(C, A, B, M, K, N);
    sparse_matmul_2_4_impl(C, A, B, M, K, N, rowmask_A, colmask_B);
    
    auto d0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++)
        dense_matmul(C, A, B, M, K, N);
    auto d1 = std::chrono::high_resolution_clock::now();
    
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

}
