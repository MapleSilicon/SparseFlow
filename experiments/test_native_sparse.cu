#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>

// Your existing PTX kernel
__device__ __forceinline__ void mma_sp_sync_m16n8k16(
    float* d, const uint32_t* a, const uint32_t* b,
    const float* c, uint32_t e, int trans_a, int trans_b
) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "%14, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e)
    );
}

// Simple test kernel
__global__ void test_sparse_kernel(
    const half* A, const half* Bc, const uint32_t* E,
    float* C, int M, int N, int K
) {
    int row = blockIdx.y * 16 + (threadIdx.x / 32) * 16;
    int col = blockIdx.x * 16;
    
    if (row >= M || col >= N) return;
    
    for (int i = threadIdx.x; i < 16 * 16; i += 256) {
        int r = row + i / 16;
        int c = col + i % 16;
        if (r < M && c < N) {
            C[r * N + c] = 0.0f;
        }
    }
}

int main() {
    printf("=== Testing Native 2:4 Sparse Kernel ===\n\n");
    
    int M = 128, N = 128, K = 128;
    
    FILE* fa = fopen("sparse_harness/data/dA.bin", "rb");
    FILE* fb = fopen("sparse_harness/data/dB.bin", "rb");
    FILE* fe = fopen("sparse_harness/data/dE.bin", "rb");
    
    if (!fa || !fb || !fe) {
        printf("Error: Cannot open files\n");
        return 1;
    }
    
    std::vector<half> hA(M * K);
    std::vector<half> hB(M * K / 2);
    std::vector<unsigned char> hE(M * K / 8);
    std::vector<float> hC(M * N);
    
    fread(hA.data(), 2, M * K, fa);
    fread(hB.data(), 2, M * K / 2, fb);
    fread(hE.data(), 1, M * K / 8, fe);
    fclose(fa); fclose(fb); fclose(fe);
    
    printf("✓ Loaded test data\n");
    printf("  A: %zu bytes\n", hA.size() * 2);
    printf("  B (compressed): %zu bytes\n", hB.size() * 2);
    printf("  E (metadata): %zu bytes\n", hE.size());
    
    printf("\n✓ Test infrastructure works!\n");
    printf("\nNext: Integrate full sparse_24_wmma.cu kernel\n");
    
    return 0;
}
