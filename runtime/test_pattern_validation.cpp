#include "sparseflow_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>

void test_pattern(int N, int M, const char* name) {
    std::cout << "\nTesting " << name << " pattern:" << std::endl;
    
    int rows = M * 4;  // 4 blocks
    int cols = 8;
    std::vector<float> tensor(rows * cols, 0.0f);
    
    // Create valid N:M pattern
    for (int i = 0; i < rows; i += M) {
        for (int j = 0; j < cols; j++) {
            // Put N non-zeros in each M-element block
            for (int bi = 0; bi < N && bi < M; bi++) {
                tensor[(i + bi) * cols + j] = 1.0f;
            }
        }
    }
    
    // Validate
    bool valid = validate_nm_pattern(tensor.data(), rows, cols, N, M, 1e-6f);
    std::cout << "  Quick validation: " << (valid ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    // Detailed validation
    auto result = validate_nm_pattern_detailed(tensor.data(), rows, cols, N, M, 1e-6f);
    std::cout << "  Total blocks: " << result.total_blocks << std::endl;
    std::cout << "  Invalid blocks: " << result.invalid_blocks << std::endl;
    std::cout << "  Avg nonzeros/block: " << result.actual_nonzeros_per_block << std::endl;
    std::cout << "  Detailed validation: " << (result.is_valid ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    // Test with invalid pattern
    tensor[M * cols + 0] = 1.0f;  // Add extra non-zero
    valid = validate_nm_pattern(tensor.data(), rows, cols, N, M, 1e-6f);
    std::cout << "  Invalid pattern detection: " << (!valid ? "✓ PASS" : "✗ FAIL") << std::endl;
}

int main() {
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   SparseFlow N:M Pattern Validation Tests" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    
    test_pattern(1, 4, "1:4");
    test_pattern(2, 4, "2:4");
    test_pattern(2, 8, "2:8");
    test_pattern(4, 16, "4:16");
    test_pattern(8, 32, "8:32");
    
    std::cout << "\n✅ All validation tests complete!" << std::endl;
    return 0;
}
