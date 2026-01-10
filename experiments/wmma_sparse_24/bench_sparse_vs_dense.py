import ctypes
import time
import torch
import numpy as np

# Load libraries
dense_lib = ctypes.CDLL("../wmma_dense/dense_wmma_grid.so")
sparse_lib = ctypes.CDLL("./libsparse_24.so")
utils_lib = ctypes.CDLL("./libsparse_utils.so")

dense_lib.launch_dense_wmma_fp32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

sparse_lib.launch_sparse_24_wmma.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

utils_lib.cpu_generate_24_sparse.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p
]

torch.set_grad_enabled(False)

def benchmark(M, N, K):
    print(f"\n{'='*60}")
    print(f"Matrix Size: {M}×{K} @ {K}×{N} = {M}×{N}")
    print(f"{'='*60}")
    
    # Create dense matrices
    A = torch.randn(M, K, device="cpu", dtype=torch.float16)
    B = torch.randn(K, N, device="cpu", dtype=torch.float16)
    
    # Generate 2:4 sparse pattern for B
    Bc = torch.zeros(K, N//2, dtype=torch.float16)
    E = torch.zeros((K//16) * (N//8), dtype=torch.uint32)
    
    utils_lib.cpu_generate_24_sparse(
        B.data_ptr(), K, N,
        Bc.data_ptr(), E.data_ptr()
    )
    
    # Move to GPU
    A_gpu = A.cuda()
    B_gpu = B.cuda()
    B_col = B_gpu.t().contiguous()
    Bc_gpu = Bc.cuda()
    E_gpu = E.cuda()
    
    C_dense = torch.empty(M, N, device="cuda", dtype=torch.float32)
    C_sparse = torch.empty(M, N, device="cuda", dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        dense_lib.launch_dense_wmma_fp32(
            A_gpu.data_ptr(), B_col.data_ptr(), C_dense.data_ptr(),
            M, N, K
        )
        sparse_lib.launch_sparse_24_wmma(
            A_gpu.data_ptr(), Bc_gpu.data_ptr(), E_gpu.data_ptr(),
            C_sparse.data_ptr(), M, N, K
        )
    torch.cuda.synchronize()
    
    # Benchmark dense
    iters = 20
    t0 = time.time()
    for _ in range(iters):
        dense_lib.launch_dense_wmma_fp32(
            A_gpu.data_ptr(), B_col.data_ptr(), C_dense.data_ptr(),
            M, N, K
        )
    torch.cuda.synchronize()
    dense_time = (time.time() - t0) / iters
    
    # Benchmark sparse
    t0 = time.time()
    for _ in range(iters):
        sparse_lib.launch_sparse_24_wmma(
            A_gpu.data_ptr(), Bc_gpu.data_ptr(), E_gpu.data_ptr(),
            C_sparse.data_ptr(), M, N, K
        )
    torch.cuda.synchronize()
    sparse_time = (time.time() - t0) / iters
    
    # Calculate metrics
    flops = 2 * M * N * K
    dense_tflops = flops / (dense_time * 1e12)
    sparse_tflops = flops / (sparse_time * 1e12)
    speedup = dense_time / sparse_time
    
    print(f"\nDense WMMA:")
    print(f"  Time:     {dense_time*1000:.3f} ms")
    print(f"  TFLOPS:   {dense_tflops:.2f}")
    
    print(f"\n2:4 Sparse WMMA:")
    print(f"  Time:     {sparse_time*1000:.3f} ms")
    print(f"  TFLOPS:   {sparse_tflops:.2f}")
    print(f"  Speedup:  {speedup:.2f}×")
    
    print(f"\nMemory Savings:")
    print(f"  Original B: {K*N*2/1024/1024:.2f} MB")
    print(f"  Compressed: {K*N/1024/1024:.2f} MB (50% reduction)")

if __name__ == "__main__":
    print("SparseFlow 2:4 Structured Sparsity Benchmark")
    print("=" * 60)
    
    benchmark(512, 512, 512)
    benchmark(1024, 1024, 1024)
    benchmark(2048, 2048, 2048)
    benchmark(4096, 4096, 4096)
