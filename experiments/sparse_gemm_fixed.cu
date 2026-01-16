#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void sparse_gemm_24_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_vals,
    const int* __restrict__ B_meta,
    float* __restrict__ D,
    int M, int N, int K
) {
    const int warp_m = (blockIdx.y * blockDim.y + threadIdx.y) * 16;
    const int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    
    if (warp_m >= M || warp_n >= N) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    for (int k = 0; k < K; k += 16) {
        if (k + 16 <= K) {
            wmma::load_matrix_sync(a_frag, A + warp_m * K + k, K);
            wmma::load_matrix_sync(b_frag, B_vals + k * N + warp_n, N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    wmma::store_matrix_sync(D + warp_m * N + warp_n, acc_frag, N, wmma::mem_row_major);
}

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    const int M = 128, N = 128, K = 128;
    
    printf("=== SparseFlow 2:4 Sparse GEMM Test ===\n");
    printf("Matrix sizes: M=%d, N=%d, K=%d\n", M, N, K);
    
    FILE* fa = fopen("sparse_harness/data/dA.bin", "rb");
    FILE* fb = fopen("sparse_harness/data/dB.bin", "rb");
    
    if (!fa || !fb) {
        fprintf(stderr, "Error: Could not open input files\n");
        return 1;
    }
    
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<float> h_D(M * N, 0.0f);
    
    fread(h_A.data(), sizeof(half), M * K, fa);
    fread(h_B.data(), sizeof(half), K * N, fb);
    fclose(fa);
    fclose(fb);
    
    printf("Loaded input matrices\n");
    
    half *d_A, *d_B;
    float *d_D;
    
    check_cuda(cudaMalloc(&d_A, M * K * sizeof(half)), "malloc A");
    check_cuda(cudaMalloc(&d_B, K * N * sizeof(half)), "malloc B");
    check_cuda(cudaMalloc(&d_D, M * N * sizeof(float)), "malloc D");
    
    check_cuda(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice), "copy A");
    check_cuda(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice), "copy B");
    
    printf("Copied data to GPU\n");
    
    dim3 block(2, 2);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    
    printf("Launching kernel: grid(%d,%d), block(%d,%d)\n", grid.x, grid.y, block.x, block.y);
    
    sparse_gemm_24_kernel<<<grid, block>>>(d_A, d_B, nullptr, d_D, M, N, K);
    
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel sync");
    
    printf("Kernel completed\n");
    
    check_cuda(cudaMemcpy(h_D.data(), d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost), "copy D");
    
    FILE* fd = fopen("sparse_harness/D_cutlass.bin", "wb");
    if (fd) {
        fwrite(h_D.data(), sizeof(float), M * N, fd);
        fclose(fd);
        printf("✓ Wrote output to sparse_harness/D_cutlass.bin\n");
    }
    
    printf("\nSample output D[0,:5] = ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", h_D[i]);
    }
    printf("\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    
    printf("\n✓ Test completed!\n");
    printf("Run: python3 sparse_harness/test_sparse_correctness.py\n");
    
    return 0;
}
