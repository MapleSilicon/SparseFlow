import torch
import time
import ctypes
import numpy as np

# Import compression utility
from compress_24 import compress_24_weights

# Load libraries
dense_lib = ctypes.CDLL("../wmma_dense/dense_wmma_grid.so")
sparse_lib = ctypes.CDLL("./libsparse_ptx_v3.so")

dense_lib.launch_dense_wmma_fp32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

sparse_lib.launch_sparse_ptx_v3.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

torch.manual_seed(42)
torch.set_grad_enabled(False)

def benchmark(M, N, K):
    print(f"\n{'='*70}")
    print(f"Matrix Size: {M}×{K} @ {K}×{N} = {M}×{N}")
    print(f"{'='*70}")
    
    # Create matrices
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    
    # Compress A to 2:4 format
    print("Compressing A to 2:4 sparse format...")
    A_sp, E = compress_24_weights(A)
    print(f"  Original A: {A.shape} = {A.numel() * 2 / 1024**2:.2f} MB")
    print(f"  Compressed: {A_sp.shape} = {A_sp.numel() * 2 / 1024**2:.2f} MB (50% reduction)")
    print(f"  Metadata:   {E.shape} = {E.numel() * 4 / 1024**2:.2f} MB")
    
    # Prepare outputs
    B_col = B.t().contiguous()
    C_dense = torch.empty(M, N, device="cuda", dtype=torch.float32)
    C_sparse = torch.empty(M, N, device="cuda", dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        dense_lib.launch_dense_wmma_fp32(
            A.data_ptr(), B_col.data_ptr(), C_dense.data_ptr(), M, N, K
        )
        sparse_lib.launch_sparse_ptx_v3(
            A_sp.data_ptr(), E.data_ptr(), B.data_ptr(), 
            C_sparse.data_ptr(), M, N, K
        )
    torch.cuda.synchronize()
    
    # Benchmark dense
    iters = 20
    t0 = time.time()
    for _ in range(iters):
        dense_lib.launch_dense_wmma_fp32(
            A.data_ptr(), B_col.data_ptr(), C_dense.data_ptr(), M, N, K
        )
    torch.cuda.synchronize()
    dense_time = (time.time() - t0) / iters
    
    # Benchmark sparse
    t0 = time.time()
    for _ in range(iters):
        sparse_lib.launch_sparse_ptx_v3(
            A_sp.data_ptr(), E.data_ptr(), B.data_ptr(),
            C_sparse.data_ptr(), M, N, K
        )
    torch.cuda.synchronize()
    sparse_time = (time.time() - t0) / iters
    
    # Calculate metrics
    flops = 2 * M * N * K
    dense_tflops = flops / (dense_time * 1e12)
    sparse_tflops = flops / (sparse_time * 1e12)
    speedup = dense_time / sparse_time
    
    print(f"\n{'Results':-^70}")
    print(f"Dense WMMA:          {dense_time*1000:6.3f} ms  |  {dense_tflops:6.2f} TFLOPS")
    print(f"Sparse PTX v3.0:     {sparse_time*1000:6.3f} ms  |  {sparse_tflops:6.2f} TFLOPS")
    print(f"{'='*70}")
    print(f"Speedup:             {speedup:.2f}×")
    print(f"Memory Savings:      50% (compressed A)")
    
    if sparse_tflops > 200:
        print(f"\n🎉 BREAKTHROUGH: {sparse_tflops:.2f} TFLOPS > 200 TFLOPS TARGET!")

if __name__ == "__main__":
    print("="*70)
    print("SparseFlow PTX v3.0: 2:4 Structured Sparsity")
    print("Target: Break 200 TFLOPS barrier")
    print("="*70)
    
    # Progressive benchmarks
    benchmark(512, 512, 512)
    benchmark(1024, 1024, 1024)
    benchmark(2048, 2048, 2048)
    benchmark(4096, 4096, 4096)
