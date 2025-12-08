#ifndef SPARSEFLOW_RUNTIME_H
#define SPARSEFLOW_RUNTIME_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/// Sparse 2:4 MatMul Kernel (CPU / OpenMP)
/// 
/// out[M×N] = lhs[M×K] × rhs[K×N]
/// rowmask[M] — 1 = row active, 0 = row skipped
/// colmask[N] — 1 = col active, 0 = col skipped
///
/// ABI is fully explicit for JIT stability.
void sparse_matmul_2_4(
    float* out,
    const float* lhs,
    const float* rhs,
    int64_t M,
    int64_t K,
    int64_t N,
    const int64_t* rowmask,
    const int64_t* colmask
);

#ifdef __cplusplus
}
#endif

#endif // SPARSEFLOW_RUNTIME_H
