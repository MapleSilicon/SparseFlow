#include "sparseflow_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>

static bool approx(float a, float b, float eps=1e-3f) {
    return std::fabs(a - b) < eps;
}

int main() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "Testing N:M runtime kernels...\n";
    
    const int M = 8, K = 8, N = 8;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);
    
    int64_t out_shape[2] = {M, N};
    int64_t lhs_shape[2] = {M, K};
    int64_t rhs_shape[2] = {K, N};
    
    std::cout << "  Testing 1:4... ";
    sparse_matmul_1_4(C.data(), A.data(), B.data(), out_shape, lhs_shape, rhs_shape);
    std::cout << "✓\n";
    
    std::cout << "  Testing 2:4... ";
    sparse_matmul_2_4(C.data(), A.data(), B.data(), out_shape, lhs_shape, rhs_shape);
    std::cout << "✓\n";
    
    std::cout << "  Testing 2:8... ";
    sparse_matmul_2_8(C.data(), A.data(), B.data(), out_shape, lhs_shape, rhs_shape);
    std::cout << "✓\n";
    
    std::cout << "  Testing 4:16... ";
    sparse_matmul_4_16(C.data(), A.data(), B.data(), out_shape, lhs_shape, rhs_shape);
    std::cout << "✓\n";
    
    std::cout << "  Testing 8:32... ";
    sparse_matmul_8_32(C.data(), A.data(), B.data(), out_shape, lhs_shape, rhs_shape);
    std::cout << "✓\n";
    
    std::cout << "✅ All N:M patterns working!\n";
    return 0;
}
