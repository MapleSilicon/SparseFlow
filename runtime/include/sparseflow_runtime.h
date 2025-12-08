//===- sparseflow_runtime.h - SparseFlow Runtime API ---------------------===//
#ifndef SPARSEFLOW_RUNTIME_H
#define SPARSEFLOW_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparseMatrixHandle* SparseMatrix;

SparseMatrix sparseflow_create_matrix(
    const bool* rowmask, int rows,
    const bool* colmask, int cols);

void sparseflow_destroy_matrix(SparseMatrix mat);

void sparse_matmul_2_4_impl(
    float* C, const float* A, const float* B,
    int M, int K, int N,
    const bool* rowmask_A, const bool* colmask_B);

void sparse_matmul_2_4(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape);

typedef struct {
    double dense_time_ms;
    double sparse_time_ms;
    double speedup;
    int64_t dense_flops;
    int64_t sparse_flops;
} BenchmarkResult;

BenchmarkResult sparseflow_benchmark_matmul(
    const float* A, const float* B,
    int M, int K, int N,
    const bool* rowmask_A, const bool* colmask_B,
    int num_iterations);

#ifdef __cplusplus
}
#endif

#endif
