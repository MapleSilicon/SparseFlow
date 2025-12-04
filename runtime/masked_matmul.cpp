#include "masked_matmul.h"
#include <omp.h>
#include <algorithm>
#include <cstring>

static const int TILE = 64;   // cache-friendly tile size

// Dense baseline for comparison
void dense_matmul(const float *A, const float *B, float *C,
                  int M, int N, int K, size_t &flops) {
    flops = 2ULL * M * N * K;
    
    std::memset(C, 0, M * N * sizeof(float));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Sparse matmul with std::vector interface (for benchmark)
void sparseflow_masked_matmul(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    const std::vector<uint8_t> &rowmask,
    const std::vector<uint8_t> &colmask,
    size_t &flops)
{
    // Count active elements
    size_t active_rows = 0, active_cols = 0;
    for (int i = 0; i < M; i++) if (rowmask[i]) active_rows++;
    for (int j = 0; j < N; j++) if (colmask[j]) active_cols++;
    
    flops = 2ULL * active_rows * active_cols * K;
    
    // Zero C
    std::memset(C, 0, M * N * sizeof(float));
    
    // Blocked matmul
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t ii = 0; ii < M; ii += TILE) {
        for (int64_t jj = 0; jj < N; jj += TILE) {

            // If this entire tile is masked out → skip
            bool row_active = false;
            for (int64_t i = ii; i < std::min<int64_t>(ii + TILE, M); i++)
                if (rowmask[i]) { row_active = true; break; }
            if (!row_active) continue;

            bool col_active = false;
            for (int64_t j = jj; j < std::min<int64_t>(jj + TILE, N); j++)
                if (colmask[j]) { col_active = true; break; }
            if (!col_active) continue;

            // Compute tile
            for (int64_t i = ii; i < std::min<int64_t>(ii + TILE, M); i++) {

                if (!rowmask[i]) continue; // skip full row

                for (int64_t j = jj; j < std::min<int64_t>(jj + TILE, N); j++) {

                    if (!colmask[j]) continue; // skip column

                    float sum = 0.0f;

                    // Inner block over K
                    for (int64_t k = 0; k < K; k++) {
                        sum += A[i * K + k] * B[k * N + j];
                    }

                    C[i * N + j] += sum;
                }
            }
        }
    }
}

// C-style runtime interface (for MLIR lowering)
extern "C" void sparseflow_masked_matmul_runtime(
    const float *A,
    const float *B,
    float *C,
    const uint8_t *rowmask,
    const uint8_t *colmask,
    int64_t M,
    int64_t N,
    int64_t K)
{
    // Zero C
    std::memset(C, 0, M * N * sizeof(float));
    
    // Blocked matmul
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t ii = 0; ii < M; ii += TILE) {
        for (int64_t jj = 0; jj < N; jj += TILE) {

            // If this entire tile is masked out → skip
            bool row_active = false;
            for (int64_t i = ii; i < std::min(ii + TILE, M); i++)
                if (rowmask[i]) { row_active = true; break; }
            if (!row_active) continue;

            bool col_active = false;
            for (int64_t j = jj; j < std::min(jj + TILE, N); j++)
                if (colmask[j]) { col_active = true; break; }
            if (!col_active) continue;

            // Compute tile
            for (int64_t i = ii; i < std::min(ii + TILE, M); i++) {

                if (!rowmask[i]) continue; // skip full row

                for (int64_t j = jj; j < std::min(jj + TILE, N); j++) {

                    if (!colmask[j]) continue; // skip column

                    float sum = 0.0f;

                    // Inner block over K
                    for (int64_t k = 0; k < K; k++) {
                        sum += A[i * K + k] * B[k * N + j];
                    }

                    C[i * N + j] += sum;
                }
            }
        }
    }
}
