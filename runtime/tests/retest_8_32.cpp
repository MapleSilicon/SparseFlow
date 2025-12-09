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
    const int SIZE = 1024;
    const int RUNS = 5;
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     8:32 Pattern Stability Test (5 runs, 1024×1024)           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    std::vector<float> A(SIZE * SIZE, 1.0f);
    std::vector<float> B(SIZE * SIZE, 1.0f);
    std::vector<float> C(SIZE * SIZE, 0.0f);
    
    int64_t shapes[2] = {SIZE, SIZE};
    
    std::vector<double> dense_times;
    std::vector<double> sparse_times;
    
    for (int run = 0; run < RUNS; run++) {
        std::cout << "Run " << (run + 1) << "/" << RUNS << "...\n";
        
        // Warmup
        dense_matmul(C.data(), A.data(), B.data(), SIZE, SIZE, SIZE);
        sparse_matmul_8_32(C.data(), A.data(), B.data(), shapes, shapes, shapes);
        
        // Dense
        auto t0 = Clock::now();
        for (int i = 0; i < 3; i++) {
            dense_matmul(C.data(), A.data(), B.data(), SIZE, SIZE, SIZE);
        }
        auto t1 = Clock::now();
        double dense_ms = Ms(t1 - t0).count() / 3.0;
        dense_times.push_back(dense_ms);
        
        // Sparse
        auto t2 = Clock::now();
        for (int i = 0; i < 3; i++) {
            sparse_matmul_8_32(C.data(), A.data(), B.data(), shapes, shapes, shapes);
        }
        auto t3 = Clock::now();
        double sparse_ms = Ms(t3 - t2).count() / 3.0;
        sparse_times.push_back(sparse_ms);
        
        std::cout << "  Dense: " << std::fixed << std::setprecision(2) << dense_ms 
                  << "ms, Sparse: " << sparse_ms << "ms, Speedup: " 
                  << (dense_ms / sparse_ms) << "×\n";
    }
    
    // Calculate median
    std::sort(dense_times.begin(), dense_times.end());
    std::sort(sparse_times.begin(), sparse_times.end());
    
    double median_dense = dense_times[RUNS / 2];
    double median_sparse = sparse_times[RUNS / 2];
    double median_speedup = median_dense / median_sparse;
    
    std::cout << "\n";
    std::cout << "┌─────────────────────────┬──────────────┐\n";
    std::cout << "│ Metric                  │ Median Value │\n";
    std::cout << "├─────────────────────────┼──────────────┤\n";
    std::cout << "│ Dense (ms)              │ " << std::setw(12) << std::fixed << std::setprecision(2) << median_dense << " │\n";
    std::cout << "│ Sparse (ms)             │ " << std::setw(12) << median_sparse << " │\n";
    std::cout << "│ Speedup                 │ " << std::setw(10) << median_speedup << "× │\n";
    std::cout << "└─────────────────────────┴──────────────┘\n";
    std::cout << "\n";
    
    if (median_speedup > 25) {
        std::cout << "✅ 8:32 pattern shows stable >25× speedup!\n";
    } else if (median_speedup > 15) {
        std::cout << "✅ 8:32 pattern shows solid ~" << (int)median_speedup << "× speedup (consistent with other patterns)\n";
    } else {
        std::cout << "⚠️  Lower than expected, may need investigation\n";
    }
    
    std::cout << "\n";
    return 0;
}
