import torch
import time
import ctypes
import numpy as np

print("="*70)
print("Benchmarking: Reference v3 vs Minimal TC Kernel")
print("="*70)

# Load libraries
ref_lib = ctypes.CDLL("./libsparse_ref_v3.so")
tc_lib = ctypes.CDLL("./libsparse_tc_minimal.so")

ref_lib.launch_sparse_ref_v3.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

tc_lib.launch_sparse_tc_minimal.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

M = N = K = 128

# Load data
Acomp = np.fromfile("dA.bin", np.float16).reshape(M, K//2)
E = np.fromfile("dE.bin", np.uint16)
B = np.fromfile("dB.bin", np.float16).reshape(K, N)

A_t = torch.from_numpy(Acomp).cuda()
E_t = torch.from_numpy(E).cuda()
B_t = torch.from_numpy(B).cuda()
C_ref = torch.empty(M, N, device='cuda', dtype=torch.float32)
C_tc = torch.empty(M, N, device='cuda', dtype=torch.float32)

# Warmup
for _ in range(20):
    ref_lib.launch_sparse_ref_v3(
        A_t.data_ptr(), E_t.data_ptr(), B_t.data_ptr(), C_ref.data_ptr(), M, N, K
    )
    tc_lib.launch_sparse_tc_minimal(
        A_t.data_ptr(), E_t.data_ptr(), B_t.data_ptr(), C_tc.data_ptr(), M, N, K
    )
torch.cuda.synchronize()

# Benchmark reference
iters = 1000
torch.cuda.synchronize()
t0 = time.time()
for _ in range(iters):
    ref_lib.launch_sparse_ref_v3(
        A_t.data_ptr(), E_t.data_ptr(), B_t.data_ptr(), C_ref.data_ptr(), M, N, K
    )
torch.cuda.synchronize()
ref_time = (time.time() - t0) / iters

# Benchmark minimal TC
torch.cuda.synchronize()
t0 = time.time()
for _ in range(iters):
    tc_lib.launch_sparse_tc_minimal(
        A_t.data_ptr(), E_t.data_ptr(), B_t.data_ptr(), C_tc.data_ptr(), M, N, K
    )
torch.cuda.synchronize()
tc_time = (time.time() - t0) / iters

# Metrics
flops = 2 * M * N * K
ref_tflops = flops / (ref_time * 1e12)
tc_tflops = flops / (tc_time * 1e12)
speedup = ref_time / tc_time

print(f"\nShape: {M}×{K} @ {K}×{N} = {M}×{N}")
print(f"{'Results':-^70}")
print(f"Reference v3:  {ref_time*1000:7.4f} ms  |  {ref_tflops:6.2f} TFLOPS")
print(f"Minimal TC:    {tc_time*1000:7.4f} ms  |  {tc_tflops:6.2f} TFLOPS")
print(f"{'='*70}")
print(f"Speedup:       {speedup:.3f}×")
print(f"\n💡 Next: Add tensor core instructions (mma.sp.sync)")
