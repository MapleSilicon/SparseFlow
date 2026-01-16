#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void sparse_gemm_24(
    const half* A_compressed,
    const half* B,
    const uint8_t* E,
    float* D,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    int K_HALF = K / 2;
    
    for (int k_group = 0; k_group < K / 4; k_group++) {
        int compressed_base = row * K_HALF + k_group * 2;
        
        int meta_byte_idx = (row * K + k_group * 4) / 16;
        int meta_shift = ((row * K + k_group * 4) / 4) % 4 * 2;
        uint8_t meta = (E[meta_byte_idx] >> meta_shift) & 0x3;
        
        int pos0, pos1;
        switch(meta) {
            case 0: pos0 = 0; pos1 = 1; break;
            case 1: pos0 = 0; pos1 = 2; break;
            case 2: pos0 = 0; pos1 = 3; break;
            case 3: pos0 = 1; pos1 = 2; break;
            default: pos0 = 0; pos1 = 1; break;
        }
        
        half val0 = A_compressed[compressed_base + 0];
        half val1 = A_compressed[compressed_base + 1];
        
        int k0 = k_group * 4 + pos0;
        int k1 = k_group * 4 + pos1;
        
        sum += __half2float(val0) * __half2float(B[col * K + k0]);
        sum += __half2float(val1) * __half2float(B[col * K + k1]);
    }
    
    D[row * N + col] = sum;
}

int main() {
    const int M = 128, N = 128, K = 128;
    const int K_HALF = K / 2;
    
    printf("=== 2:4 Sparse GEMM ===\n");
    
    FILE* fa = fopen("dA.bin", "rb");
    FILE* fb = fopen("dB.bin", "rb");
    FILE* fe = fopen("dE.bin", "rb");
    
    if (!fa || !fb || !fe) {
        fprintf(stderr, "Error: Missing files\n");
        return 1;
    }
    
    std::vector<half> h_A(M * K_HALF);
    std::vector<half> h_B(K * N);
    std::vector<uint8_t> h_E(M * K / 8);
    std::vector<float> h_D(M * N);
    
    fread(h_A.data(), sizeof(half), M * K_HALF, fa);
    fread(h_B.data(), sizeof(half), K * N, fb);
    fread(h_E.data(), 1, M * K / 8, fe);
    fclose(fa); fclose(fb); fclose(fe);
    
    half *d_A, *d_B;
    uint8_t *d_E;
    float *d_D;
    
    cudaMalloc(&d_A, h_A.size() * sizeof(half));
    cudaMalloc(&d_B, h_B.size() * sizeof(half));
    cudaMalloc(&d_E, h_E.size());
    cudaMalloc(&d_D, h_D.size() * sizeof(float));
    
    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E.data(), h_E.size(), cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    sparse_gemm_24<<<grid, block>>>(d_A, d_B, d_E, d_D, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_D.data(), d_D, h_D.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* fd = fopen("D_cutlass.bin", "wb");
    fwrite(h_D.data(), sizeof(float), M * N, fd);
    fclose(fd);
    
    printf("Done!\n");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_E); cudaFree(d_D);
    return 0;
}
