#include "sparseflow_runtime.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "   SparseFlow N:M Pattern Validation Tests\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    // Create test matrices
    const int rows = 32, cols = 8;
    std::vector<float> valid_1_4(rows * cols, 0.0f);
    std::vector<float> valid_2_4(rows * cols, 0.0f);
    
    // Fill with valid 1:4 pattern (1 non-zero per 4 rows)
    for (int i = 0; i < rows; i += 4) {
        for (int j = 0; j < cols; j++) {
            valid_1_4[i * cols + j] = 1.0f;
        }
    }
    
    // Fill with valid 2:4 pattern (2 non-zeros per 4 rows)
    for (int i = 0; i < rows; i += 4) {
        for (int j = 0; j < cols; j++) {
            valid_2_4[i * cols + j] = 1.0f;
            valid_2_4[(i+1) * cols + j] = 1.0f;
        }
    }
    
    std::cout << "Testing 1:4 pattern:\n";
    if (validate_nm_pattern(valid_1_4.data(), rows, cols, 1, 4, 1e-6f)) {
        std::cout << "  Quick validation: ✓ PASS\n";
    } else {
        std::cout << "  Quick validation: ✗ FAIL\n";
    }
    
    auto result = validate_nm_pattern_detailed(valid_1_4.data(), rows, cols, 1, 4, 1e-6f);
    std::cout << "  Total blocks: " << result.total_blocks << "\n";
    std::cout << "  Invalid blocks: " << result.invalid_blocks << "\n";
    std::cout << "  Avg nonzeros/block: " << result.actual_nonzeros_per_block << "\n";
    if (result.is_valid) {
        std::cout << "  Detailed validation: ✓ PASS\n";
    } else {
        std::cout << "  Detailed validation: ✗ FAIL\n";
    }
    
    std::cout << "Testing 2:4 pattern:\n";
    if (validate_nm_pattern(valid_2_4.data(), rows, cols, 2, 4, 1e-6f)) {
        std::cout << "  Quick validation: ✓ PASS\n";
    } else {
        std::cout << "  Quick validation: ✗ FAIL\n";
    }
    
    result = validate_nm_pattern_detailed(valid_2_4.data(), rows, cols, 2, 4, 1e-6f);
    std::cout << "  Total blocks: " << result.total_blocks << "\n";
    std::cout << "  Invalid blocks: " << result.invalid_blocks << "\n";
    std::cout << "  Avg nonzeros/block: " << result.actual_nonzeros_per_block << "\n";
    if (result.is_valid) {
        std::cout << "  Detailed validation: ✓ PASS\n";
    } else {
        std::cout << "  Detailed validation: ✗ FAIL\n";
    }
    
    std::cout << "✅ All validation tests complete!\n";
    return 0;
}
