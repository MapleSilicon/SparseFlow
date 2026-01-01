#!/usr/bin/env python3
import torch
import ctypes
import time

lib = ctypes.CDLL('./gemm_fused_relu_v1.so')
lib.launch_gemm_fused_relu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3

def benchmark(M):
    A = torch.randn(M, M, device='cuda', dtype=torch.float16)
    B = torch.randn(M, M, device='cuda', dtype=torch.float16)
    C = torch.zeros(M, M, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        lib.launch_gemm_fused_relu(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, M, M)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    iterations = 50
    for _ in range(iterations):
        lib.launch_gemm_fused_relu(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, M, M)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    tflops = (2 * M**3 * iterations) / elapsed / 1e12
    return tflops

if __name__ == '__main__':
    print("SparseFlow v1.0 Benchmark")
    print("=" * 50)
    
    for size in [2048, 4096, 8192]:
        tflops = benchmark(size)
        vs_cublas = (tflops / 21) * 100
        print(f"{size}×{size:4d}  |  {tflops:5.2f} TFLOPS  |  {vs_cublas:5.1f}%")
