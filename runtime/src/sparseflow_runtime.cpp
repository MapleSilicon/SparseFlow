//===----------------------------------------------------------------------===//
// SparseFlow Runtime — 2:4 Structured Sparse MatMul (CPU / OpenMP)
//===----------------------------------------------------------------------===//

#include "sparseflow_runtime.h"
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

/// Runtime implementation of sparse 2:4 matmul.
/// This version correctly handles rowmask / colmask arrays.
void sparse_matmul_2_4(
    float* out,
    const float* lhs,
    const float* rhs,
    int64_t M,
    int64_t K,
    int64_t N,
    const int64_t* rowmask,
    const int64_t* colmask
) {
    // Zero output
    memset(out, 0, sizeof(float) * M * N);

    // Parallel outer loop
    #pragma omp parallel for
    for (int64_t i = 0; i < M; i++) {
        if (rowmask && rowmask[i] == 0)
            continue;

        for (int64_t j = 0; j < N; j++) {
            if (colmask && colmask[j] == 0)
                continue;

            float acc = 0.0f;

            // Dense inner loop (compute only active rows/cols)
            for (int64_t p = 0; p < K; p++)
                acc += lhs[i*K + p] * rhs[p*N + j];

            out[i*N + j] = acc;
        }
    }
}

} // extern "C"
