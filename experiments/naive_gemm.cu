#include <cstdio>
#include <vector>
#include <cuda_fp16.h>

__global__ void gemm_kernel(
    const half* A,
    const half* B,
    float* D,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += __half2float(A[row * K + k]) *
                   __half2float(B[col * K + k]); // B is column-major
        }
        D[row * N + col] = acc;
    }
}

int main() {
    const int M = 128, N = 128, K = 128;

    FILE* fa = fopen("A_pruned_dense.bin", "rb");
    FILE* fb = fopen("B_dense.bin", "rb");
    if (!fa || !fb) { printf("Missing input files\n"); return 1; }

    std::vector<half> hA(M*K), hB(K*N);
    fread(hA.data(), sizeof(half), M*K, fa);
    fread(hB.data(), sizeof(half), K*N, fb);
    fclose(fa); fclose(fb);

    half *dA, *dB;
    float *dD;
    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*N*sizeof(half));
    cudaMalloc(&dD, M*N*sizeof(float));

    cudaMemcpy(dA, hA.data(), M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), K*N*sizeof(half), cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((N+15)/16, (M+15)/16);
    gemm_kernel<<<grid, block>>>(dA, dB, dD, M, N, K);
    cudaDeviceSynchronize();

    std::vector<float> hD(M*N);
    cudaMemcpy(hD.data(), dD, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fd = fopen("D_naive_gpu.bin", "wb");
    fwrite(hD.data(), sizeof(float), M*N, fd);
    fclose(fd);

    printf("✓ Wrote D_naive_gpu.bin\n");
    printf("Sample D[0,:5] = %.6f %.6f %.6f %.6f %.6f\n",
           hD[0], hD[1], hD[2], hD[3], hD[4]);

    return 0;
}
