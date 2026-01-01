#include "sparseflow_runtime.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

void dense_matmul(float* C, const float* A, const float* B, int M, int K, int N) {
    std::memset(C, 0, sizeof(float) * M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

int main() {
    const int SIZE = 1024; // Production size, fast enough for demos
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     SparseFlow v0.2 Quick Benchmark (1024×1024)               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    std::vector<float> A(SIZE * SIZE, 1.0f);
    std::vector<float> B(SIZE * SIZE, 1.0f);
    std::vector<float> C(SIZE * SIZE, 0.0f);
    
    int64_t shapes[2] = {SIZE, SIZE};
    
    // Warmup
    dense_matmul(C.data(), A.data(), B.data(), SIZE, SIZE, SIZE);
    sparse_matmul_2_4(C.data(), A.data(), B.data(), shapes, shapes, shapes);
    
    std::cout << "Benchmarking 1024×1024 matrix multiplication...\n\n";
    
    // Dense
    auto t0 = Clock::now();
    for (int i = 0; i < 5; i++) {
        dense_matmul(C.data(), A.data(), B.data(), SIZE, SIZE, SIZE);
    }
    auto t1 = Clock::now();
    double dense_ms = Ms(t1 - t0).count() / 5.0;
    
    // Sparse 2:4
    auto t2 = Clock::now();
    for (int i = 0; i < 5; i++) {
        sparse_matmul_2_4(C.data(), A.data(), B.data(), shapes, shapes, shapes);
    }
    auto t3 = Clock::now();
    double sparse_ms = Ms(t3 - t2).count() / 5.0;
    
    double speedup = dense_ms / sparse_ms;
    
    std::cout << "┌─────────────────────────┬──────────────┐\n";
    std::cout << "│ Metric                  │ Value        │\n";
    std::cout << "├─────────────────────────┼──────────────┤\n";
    std::cout << "│ Dense Baseline          │ " << std::setw(9) << std::fixed << std::setprecision(2) << dense_ms << " ms │\n";
    std::cout << "│ Sparse 2:4              │ " << std::setw(9) << sparse_ms << " ms │\n";
    std::cout << "│ Speedup                 │ " << std::setw(9) << speedup << "×  │\n";
    std::cout << "│ Matrix Size             │    1024×1024 │\n";
    std::cout << "│ Operations              │      2.15 GB │\n";
    std::cout << "└─────────────────────────┴──────────────┘\n";
    std::cout << "\n";
    std::cout << "✅ Quick benchmark complete! ~" << speedup << "× faster with 2:4 sparsity\n";
    std::cout << "\n";
    
    return 0;
}
