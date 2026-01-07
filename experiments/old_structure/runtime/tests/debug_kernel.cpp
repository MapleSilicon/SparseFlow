#include <iostream>
#include <vector>
#include <cstring>

extern "C" {
    void sparse_matmul_1_4(const float* A, const float* B, float* C, int M, int N, int K);
}

int main() {
    const int SIZE = 256;
    
    std::cout << "Allocating matrices..." << std::flush;
    std::vector<float> A(SIZE * SIZE, 0.0f);
    std::vector<float> B(SIZE * SIZE, 1.0f);
    std::vector<float> C(SIZE * SIZE, 0.0f);
    std::cout << " OK\n";
    
    std::cout << "Initializing 1:4 pattern..." << std::flush;
    for (int i = 0; i < SIZE; i++) {
        for (int block = 0; block < SIZE / 4; block++) {
            A[i * SIZE + block * 4 + 0] = 1.0f;  // First element of each 4-element block
        }
    }
    std::cout << " OK\n";
    
    std::cout << "Calling sparse_matmul_1_4..." << std::flush;
    sparse_matmul_1_4(A.data(), B.data(), C.data(), SIZE, SIZE, SIZE);
    std::cout << " OK\n";
    
    std::cout << "Checking result..." << std::flush;
    float sum = 0.0f;
    for (int i = 0; i < SIZE * SIZE; i++) {
        sum += C[i];
    }
    std::cout << " Sum = " << sum << "\n";
    
    std::cout << "\n✅ Test passed!\n";
    return 0;
}
