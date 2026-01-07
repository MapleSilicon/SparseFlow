#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║    🚀 SPARSEFLOW v0.2.0 ULTIMATE PERFORMANCE TEST 🚀           ║"
echo "║                                                                ║"
echo "║         REAL MEASUREMENTS - NO TIME LIMITS                     ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="ultimate_benchmark_${TIMESTAMP}.txt"
CSV_FILE="ultimate_benchmark_${TIMESTAMP}.csv"

# Redirect all output to file AND screen
exec > >(tee -a "$RESULTS_FILE")
exec 2>&1

echo "Test started: $(date)"
echo "Results file: $RESULTS_FILE"
echo "CSV export: $CSV_FILE"
echo ""
echo "System Information:"
echo "  OS: $(uname -a)"
echo "  CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "  Cores: $(nproc)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Rebuild to ensure latest code
echo "════════════════════════════════════════════════════════════════"
echo "🔨 Step 1: Rebuilding Runtime"
echo "════════════════════════════════════════════════════════════════"
cd runtime/build
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Build successful${NC}"
echo ""

# Quick validation
echo "════════════════════════════════════════════════════════════════"
echo "🧪 Step 2: Quick Validation"
echo "════════════════════════════════════════════════════════════════"
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
./test_nm_runtime
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Validation failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Validation passed${NC}"
echo ""

# Create CSV header
echo "Pattern,MatrixSize,Dense_ms,Sparse_ms,Speedup,Density" > "../$CSV_FILE"

# The ultimate benchmark
echo "════════════════════════════════════════════════════════════════"
echo "📊 Step 3: ULTIMATE PERFORMANCE TEST"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will take HOURS! Grab coffee ☕${NC}"
echo ""
echo "Testing matrix sizes: 256, 512, 1024, 2048"
echo "Testing patterns: 1:4, 2:4, 2:8, 4:16, 8:32"
echo ""
echo "Estimated time:"
echo "  256×256:   ~2 minutes"
echo "  512×512:   ~8 minutes"
echo "  1024×1024: ~60 minutes"
echo "  2048×2048: ~8 hours"
echo ""
echo "Total: ~9-10 hours"
echo ""

read -t 10 -p "Press Enter to start (auto-starting in 10 seconds)..." || true
echo ""
echo "Starting benchmark at $(date)..."
echo ""

# Create a custom benchmark that reports progress
cat > benchmark_with_progress.cpp << 'BENCH'
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iomanip>

extern "C" {
    void sparse_matmul_1_4(const float* A, const float* B, float* C, int M, int N, int K);
    void sparse_matmul_2_4(const float* A, const float* B, float* C, int M, int N, int K);
    void sparse_matmul_2_8(const float* A, const float* B, float* C, int M, int N, int K);
    void sparse_matmul_4_16(const float* A, const float* B, float* C, int M, int N, int K);
    void sparse_matmul_8_32(const float* A, const float* B, float* C, int M, int N, int K);
}

void naive_dense_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    std::memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

double benchmark_single(
    void (*kernel)(const float*, const float*, float*, int, int, int),
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int size,
    const char* name,
    int warmup = 3,
    int runs = 10
) {
    std::cout << "    Testing " << name << "... " << std::flush;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        std::memset(C.data(), 0, size * size * sizeof(float));
        kernel(A.data(), B.data(), C.data(), size, size, size);
    }
    
    // Measure
    std::vector<double> times;
    for (int run = 0; run < runs; run++) {
        std::memset(C.data(), 0, size * size * sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        kernel(A.data(), B.data(), C.data(), size, size, size);
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        
        if ((run + 1) % 3 == 0) {
            std::cout << "." << std::flush;
        }
    }
    
    std::sort(times.begin(), times.end());
    double median = times[runs / 2];
    std::cout << " " << median << " ms\n" << std::flush;
    
    return median;
}

int main() {
    std::vector<int> sizes = {256, 512, 1024, 2048};
    
    struct Pattern {
        const char* name;
        void (*kernel)(const float*, const float*, float*, int, int, int);
        int n, m;
    };
    
    Pattern patterns[] = {
        {"1:4",   sparse_matmul_1_4,  1, 4},
        {"2:4",   sparse_matmul_2_4,  2, 4},
        {"2:8",   sparse_matmul_2_8,  2, 8},
        {"4:16",  sparse_matmul_4_16, 4, 16},
        {"8:32",  sparse_matmul_8_32, 8, 32}
    };
    
    int total_tests = sizes.size() * (1 + 5); // dense + 5 patterns per size
    int completed = 0;
    
    for (int size : sizes) {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  Matrix Size: " << std::setw(4) << size << "×" << std::setw(4) << size;
        std::cout << " - Started: " << std::flush;
        
        auto size_start = std::chrono::system_clock::now();
        std::time_t start_time = std::chrono::system_clock::to_time_t(size_start);
        std::cout << std::put_time(std::localtime(&start_time), "%H:%M:%S");
        std::cout << "        ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
        
        std::vector<float> A(size * size, 0.0f);
        std::vector<float> B(size * size, 1.0f);
        std::vector<float> C_dense(size * size, 0.0f);
        std::vector<float> C_sparse(size * size, 0.0f);
        
        std::cout << "  [" << ++completed << "/" << total_tests << "] Dense baseline:\n";
        double dense_time = benchmark_single(naive_dense_matmul, A, B, C_dense, size, "Dense");
        
        std::cout << "\n┌────────────┬───────────┬────────────┬────────────┬──────────────┐\n";
        std::cout << "│ Pattern    │ Dense (ms)│ Sparse (ms)│ Speedup    │ Density      │\n";
        std::cout << "├────────────┼───────────┼────────────┼────────────┼──────────────┤\n";
        
        for (const auto& p : patterns) {
            std::cout << "  [" << ++completed << "/" << total_tests << "] " << p.name << " pattern:\n";
            
            // Reinitialize A for this pattern
            std::fill(A.begin(), A.end(), 0.0f);
            for (int i = 0; i < size; i++) {
                for (int block = 0; block < size / p.m; block++) {
                    for (int k = 0; k < p.n; k++) {
                        A[i * size + block * p.m + k] = 1.0f;
                    }
                }
            }
            
            double sparse_time = benchmark_single(p.kernel, A, B, C_sparse, size, p.name);
            double speedup = dense_time / sparse_time;
            double density = (double)p.n / p.m * 100;
            
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "│ " << std::setw(10) << std::left << p.name
                      << " │ " << std::setw(9) << std::right << dense_time
                      << " │ " << std::setw(10) << sparse_time
                      << " │ " << std::setw(7) << speedup << " ×  "
                      << " │ " << std::setw(12) << std::right << (int)density << "%"
                      << " │\n";
            
            // Also write to CSV
            std::cout << "CSV:" << p.name << "," << size << "," << dense_time << "," 
                      << sparse_time << "," << speedup << "," << (int)density << "\n";
            
            if (&p != &patterns[4]) {
                std::cout << "├────────────┼───────────┼────────────┼────────────┼──────────────┤\n";
            }
        }
        
        std::cout << "└────────────┴───────────┴────────────┴────────────┴──────────────┘\n";
        
        auto size_end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(size_end - size_start);
        std::cout << "\n✅ " << size << "×" << size << " completed in " << elapsed.count() << " minutes\n";
    }
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                 ✅ BENCHMARK COMPLETE! ✅                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    return 0;
}
BENCH

# Compile the progress benchmark
g++ -std=c++17 -O3 benchmark_with_progress.cpp -o benchmark_ultimate -L. -lsparseflow_runtime -Wl,-rpath,.

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Progress benchmark compiled${NC}"
    echo ""
    
    # Run it!
    ./benchmark_ultimate | tee -a "../$CSV_FILE"
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "📊 RESULTS SUMMARY"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Full results: $RESULTS_FILE"
    echo "CSV data: $CSV_FILE"
    echo ""
    echo "Test completed: $(date)"
    echo ""
    
    # Extract CSV data
    cd ..
    if [ -f "$CSV_FILE" ]; then
        echo "CSV Data Preview:"
        head -20 "$CSV_FILE"
    fi
else
    echo -e "${RED}❌ Failed to compile progress benchmark${NC}"
    exit 1
fi

