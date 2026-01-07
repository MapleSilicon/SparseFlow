#pragma once
#include <vector>
#include <cstdint>

// SPA Row/Col Masked Matmul (OpenMP accelerated)

// Dense A: (M x K)
// Dense B: (K x N)
// Output C: (M x N)
//
// rowmask: size M (0/1)
// colmask: size N (0/1)
//
// Only rows with rowmask[i] == 1 and columns with colmask[j] == 1
// are computed. All other entries in C are set to 0.
//
void sparseflow_masked_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    const std::vector<uint8_t>& rowmask,
    const std::vector<uint8_t>& colmask,
    uint64_t& flops_out   // returns FLOP count
);

// Baseline: full dense matmul (for benchmarking)
void dense_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    uint64_t& flops_out
);
