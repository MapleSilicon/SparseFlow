import torch
import time
import ctypes
import numpy as np

print("="*70)
print("SparseFlow v3: Sparse Kernel vs PyTorch cuBLAS")
print("="*70)

sparse_lib = ctypes.CDLL("./libsparse_ref_v3.so")
sparse_lib.launch_sparse_ref_v3.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

M = N = K = 128

# Load validated data
Acomp = np.fromfile("dA.bin", np.float16).reshape(M, K//2)
A_pruned = np.fromfile("A_pruned_dense.bin", np.float16).reshape(M, K)
E = np.fromfile("dE.bin", np.uint16)
B = np.fromfile("dB.bin", np.float16).reshape(K, N)

# Move to GPU
A_pruned_t = torch.from_numpy(A_pruned).cuda()
Acomp_t = torch.from_numpy(Acomp).cuda()
E_t = torch.from_numpy(E).cuda()
B_t = torch.from_numpy(B).cuda()
C_sparse = torch.empty(M, N, device='cuda', dtype=torch.float32)

print(f"Shape: {M}×{K} @ {K}×{N} = {M}×{N}")
print(f"Memory: Dense={A_pruned.nbytes/1024:.1f}KB, Sparse={Acomp.nbytes/1024:.1f}KB (50% savings)\n")

# Warmup
for _ in range(20):
    _ = torch.matmul(A_pruned_t, B_t)  # Dense cuBLAS
    sparse_lib.launch_sparse_ref_v3(
        Acomp_t.data_ptr(), E_t.data_ptr(), B_t.data_ptr(), C_sparse.data_ptr(),
        M, N, K
    )
torch.cuda.synchronize()

# Benchmark dense (PyTorch cuBLAS)
iters = 1000
torch.cuda.synchronize()
t0 = time.time()
for _ in range(iters):
    C_dense = torch.matmul(A_pruned_t, B_t)
torch.cuda.synchronize()
dense_time = (time.time() - t0) / iters

# Benchmark sparse
torch.cuda.synchronize()
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
max_err = (C_dense.float() - C_sparse).abs().max().item()
mean_err = (C_dense.float() - C_sparse).abs().mean().item()

print(f"{'Results':-^70}")
print(f"Dense (cuBLAS):  {dense_time*1000:7.4f} ms  |  {dense_tflops:6.2f} TFLOPS")
print(f"Sparse (ref v3): {sparse_time*1000:7.4f} ms  |  {sparse_tflops:6.2f} TFLOPS")
print(f"{'='*70}")
print(f"Speedup:         {speedup:.3f}×")
print(f"Memory savings:  50%")
print(f"Max error:       {max_err:.2e}")
print(f"Mean error:      {mean_err:.2e}")

if max_err < 1e-4:
    print("✅ Correctness PASS")
else:
    print("❌ Correctness FAIL")

print(f"\n💡 Note: v3 is a REFERENCE kernel (simple nested loops)")
print(f"   Goal: Optimized kernel with tensor cores → 2-4× speedup")
