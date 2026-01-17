import torch
import time
import ctypes
import numpy as np

print("="*70)
print("SparseFlow v3 Benchmark - Using Validated Data")
print("="*70)

# Load libraries
dense_lib = ctypes.CDLL("../../wmma_dense/dense_wmma_grid.so")
sparse_lib = ctypes.CDLL("./libsparse_ref_v3.so")

dense_lib.launch_dense_wmma_fp32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

sparse_lib.launch_sparse_ref_v3.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

M = N = K = 128

# Load the VALIDATED data from disk (known good)
Acomp = np.fromfile("dA.bin", np.float16).reshape(M, K//2)
E = np.fromfile("dE.bin", np.uint16)
B = np.fromfile("dB.bin", np.float16).reshape(K, N)

# Also load unpruned A for dense baseline
A_pruned = np.fromfile("A_pruned_dense.bin", np.float16).reshape(M, K)

# Move to GPU
A_t = torch.from_numpy(A_pruned).cuda()
Acomp_t = torch.from_numpy(Acomp).cuda()
E_t = torch.from_numpy(E).cuda()
B_t = torch.from_numpy(B).cuda()

B_col = B_t.t().contiguous()

C_dense = torch.empty(M, N, device='cuda', dtype=torch.float32)
C_sparse = torch.empty(M, N, device='cuda', dtype=torch.float32)

print(f"Shape: {M}×{K} @ {K}×{N} = {M}×{N}")
print(f"Using validated data from disk\n")

# Warmup
for _ in range(20):
    dense_lib.launch_dense_wmma_fp32(
        A_t.data_ptr(), B_col.data_ptr(), C_dense.data_ptr(), M, N, K
    )
    sparse_lib.launch_sparse_ref_v3(
        Acomp_t.data_ptr(), E_t.data_ptr(), B_t.data_ptr(), C_sparse.data_ptr(),
        M, N, K
    )
torch.cuda.synchronize()

# Benchmark dense
iters = 1000
t0 = time.time()
for _ in range(iters):
    dense_lib.launch_dense_wmma_fp32(
        A_t.data_ptr(), B_col.data_ptr(), C_dense.data_ptr(), M, N, K
    )
torch.cuda.synchronize()
dense_time = (time.time() - t0) / iters

# Benchmark sparse
t0 = time.time()
for _ in range(iters):
    sparse_lib.launch_sparse_ref_v3(
        Acomp_t.data_ptr(), E_t.data_ptr(), B_t.data_ptr(), C_sparse.data_ptr(),
        M, N, K
    )
torch.cuda.synchronize()
sparse_time = (time.time() - t0) / iters

# Metrics
flops = 2 * M * N * K
dense_tflops = flops / (dense_time * 1e12)
sparse_tflops = flops / (sparse_time * 1e12)
speedup = dense_time / sparse_time

# Verify correctness
max_err = (C_dense - C_sparse).abs().max().item()
mean_err = (C_dense - C_sparse).abs().mean().item()

print(f"{'Results':-^70}")
print(f"Dense WMMA:     {dense_time*1000:7.4f} ms  |  {dense_tflops:6.2f} TFLOPS")
print(f"Sparse Ref v3:  {sparse_time*1000:7.4f} ms  |  {sparse_tflops:6.2f} TFLOPS")
print(f"{'='*70}")
print(f"Speedup:        {speedup:.3f}×")
print(f"Max error:      {max_err:.2e}")
print(f"Mean error:     {mean_err:.2e}")

if max_err < 1e-3:
    print("✅ Correctness PASS")
else:
    print("❌ Correctness FAIL")
