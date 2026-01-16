#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void simple_gemm_kernel(
    const half* A, const half* B, float* D,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A is row-major: A[row, k] = A[row * K + k]
            // B is column-major: B[k, col] = B[col * K + k]
            sum += __half2float(A[row * K + k]) * __half2float(B[col * K + k]);
        }
        D[row * N + col] = sum;
    }
}

int main() {
    const int M = 128, N = 128, K = 128;
    
    printf("=== Simple GEMM Test (Fixed Layout) ===\n");
    printf("M=%d, N=%d, K=%d\n\n", M, N, K);
    
    FILE* fa = fopen("sparse_harness/data/dA.bin", "rb");
    FILE* fb = fopen("sparse_harness/data/dB.bin", "rb");
    
    if (!fa || !fb) {
        fprintf(stderr, "Error: Cannot open files\n");
        return 1;
    }
    
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<float> h_D(M * N);
    
    fread(h_A.data(), sizeof(half), M * K, fa);
    fread(h_B.data(), sizeof(half), K * N, fb);
    fclose(fa);
    fclose(fb);
    
    printf("Loaded matrices from disk\n");
    
    half *d_A, *d_B;
    float *d_D;
    
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_D, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    printf("Launching kernel...\n");
    simple_gemm_kernel<<<grid, block>>>(d_A, d_B, d_D, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* fd = fopen("sparse_harness/D_cutlass.bin", "wb");
    fwrite(h_D.data(), sizeof(float), M * N, fd);
    fclose(fd);
    
    printf("✓ Output written to sparse_harness/D_cutlass.bin\n\n");
    
    printf("Sample D[0,:5] = ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", h_D[i]);
    }
    printf("\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    
    return 0;
}
