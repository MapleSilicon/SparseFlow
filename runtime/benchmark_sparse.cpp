#include "masked_matmul.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>

// Simple random float generator
void fill_random(float* data, int size) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < size; i++)
        data[i] = dist(rng);
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 768, 1024};

    // SPA pattern (from your v0.6 demo)
    std::vector<uint8_t> base_rowmask = {1,0,1,0};
    std::vector<uint8_t> base_colmask = {1,1,0,0};

    for (int N : sizes) {
        int M=N, K=N;

        std::cout << "\n=== Testing size: " << N << "x" << N << " ===\n";

        // allocate matrices
        std::vector<float> A(M*K);
        std::vector<float> B(K*N);
        std::vector<float> C_dense(M*N);
        std::vector<float> C_sparse(M*N);

        fill_random(A.data(), M*K);
        fill_random(B.data(), K*N);

        // Expand masks
        std::vector<uint8_t> rowmask(N), colmask(N);
        for (int i = 0; i < N; i++) rowmask[i] = base_rowmask[i % 4];
        for (int j = 0; j < N; j++) colmask[j] = base_colmask[j % 4];

        // --- Dense ---
        uint64_t flops_dense = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        dense_matmul(A.data(), B.data(), C_dense.data(), M, K, N, flops_dense);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dense_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- Sparse ---
        uint64_t flops_sparse = 0;
        auto t2 = std::chrono::high_resolution_clock::now();
        sparseflow_masked_matmul(A.data(), B.data(), C_sparse.data(),
                                  M, K, N, rowmask, colmask, flops_sparse);
        auto t3 = std::chrono::high_resolution_clock::now();
        double sparse_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::cout << "Dense time:   " << dense_ms  << " ms\n";
        std::cout << "Sparse time:  " << sparse_ms << " ms\n";

        double measured = dense_ms / sparse_ms;
        double theoretical = (double)flops_dense / flops_sparse;

        std::cout << "Measured speedup:   " << measured << "x\n";
        std::cout << "Theoretical speedup: " << theoretical << "x\n";
    }

    return 0;
}
