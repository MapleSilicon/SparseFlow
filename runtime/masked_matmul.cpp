#include "masked_matmul.h"
#include <omp.h>
#include <cstring>

// Dense baseline
void dense_matmul(
    const float* A, const float* B, float* C,
    int M, int K, int N,
    uint64_t& flops_out)
{
    flops_out = 0;
    std::memset(C, 0, sizeof(float) * M * N);

    #pragma omp parallel for reduction(+:flops_out)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;

            #pragma omp simd reduction(+:acc, flops_out)
            for (int k = 0; k < K; k++) {
                acc += A[i*K + k] * B[k*N + j];
                flops_out += 2;
            }

            C[i*N + j] = acc;
        }
    }
}


// Sparse masked version (SPA)
void sparseflow_masked_matmul(
    const float* A, const float* B, float* C,
    int M, int K, int N,
    const std::vector<uint8_t>& rowmask,
    const std::vector<uint8_t>& colmask,
    uint64_t& flops_out)
{
    flops_out = 0;
    std::memset(C, 0, sizeof(float) * M * N);

    #pragma omp parallel for reduction(+:flops_out)
    for (int i = 0; i < M; i++) {
        if (rowmask[i] == 0) continue;

        for (int j = 0; j < N; j++) {
            if (colmask[j] == 0) continue;

            float acc = 0.0f;

            #pragma omp simd reduction(+:acc, flops_out)
            for (int k = 0; k < K; k++) {
                acc += A[i*K + k] * B[k*N + j];
                flops_out += 2;
            }

            C[i*N + j] = acc;
        }
    }
}
