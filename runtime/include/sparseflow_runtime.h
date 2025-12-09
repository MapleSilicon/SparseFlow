#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Matrix handle (opaque pointer)
typedef struct SparseMatrixHandle* SparseMatrix;

// Benchmark result structure
struct BenchmarkResult {
    double dense_time_ms;
    double sparse_time_ms;
    double speedup;
    int64_t dense_flops;
    int64_t sparse_flops;
};

// Create sparse matrix with row/col masks
SparseMatrix sparseflow_create_matrix(
    const bool* rowmask, int rows,
    const bool* colmask, int cols);

void sparseflow_destroy_matrix(SparseMatrix mat);

// Generic N:M sparse matmul implementation
void sparse_matmul_2_4_impl(
    float* C, const float* A, const float* B,
    int M, int K, int N,
    const bool* rowmask_A, const bool* colmask_B);

// New: Specific N:M pattern kernels
void sparse_matmul_1_4(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape);

void sparse_matmul_2_4(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape);

void sparse_matmul_2_8(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape);

void sparse_matmul_4_16(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape);

void sparse_matmul_8_32(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape);

// Benchmark function
BenchmarkResult sparseflow_benchmark_matmul(
    const float* A, const float* B,
    int M, int K, int N,
    const bool* rowmask_A, const bool* colmask_B,
    int num_iterations);

#ifdef __cplusplus
}
#endif
