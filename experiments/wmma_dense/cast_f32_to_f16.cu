#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void f32_to_f16_kernel(const float* __restrict__ in, half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

extern "C" void launch_f32_to_f16(const float* in, half* out, int n) {
    int block = 256;
    int grid  = (n + block - 1) / block;
    f32_to_f16_kernel<<<grid, block>>>(in, out, n);
}
