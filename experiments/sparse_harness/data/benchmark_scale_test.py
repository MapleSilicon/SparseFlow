import torch
import time
import ctypes
import numpy as np

print("="*70)
print("SparseFlow Scale Test: 128×128 → 2048×2048")
print("="*70)

sparse_lib = ctypes.CDLL("./libsparse_ref_v3.so")
sparse_lib.launch_sparse_ref_v3.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

def benchmark_size(M, N, K, prefix, warmup=10, iters=100):
    # Load data
    A_comp = np.fromfile(f"{prefix}_Acomp.bin", np.float16).reshape(M, K//2)
    A_pruned = np.fromfile(f"{prefix}_Apruned.bin", np.float16).reshape(M, K)
    E = np.fromfile(f"{prefix}_E.bin", np.uint16)
    B = np.fromfile(f"{prefix}_B.bin", np.float16).reshape(K, N)
    D_ref = np.fromfile(f"{prefix}_Dref.bin", np.float32).reshape(M, N)
    
    # Move to GPU
    A_pruned_t = torch.from_numpy(A_pruned).cuda()
    A_comp_t = torch.from_numpy(A_comp).cuda()
    E_t = torch.from_numpy(E).cuda()
    B_t = torch.from_numpy(B).cuda()
    D_ref_t = torch.from_numpy(D_ref).cuda()
    C_sparse = torch.empty(M, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(A_pruned_t.float(), B_t.float())
        sparse_lib.launch_sparse_ref_v3(
            A_comp_t.data_ptr(), E_t.data_ptr(), B_t.data_ptr(), 
            C_sparse.data_ptr(), M, N, K
        )
    torch.cuda.synchronize()
    
    # Benchmark dense
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        C_dense = torch.matmul(A_pruned_t.float(), B_t.float())
    torch.cuda.synchronize()
    dense_time = (time.time() - t0) / iters
    
    # Benchmark sparse
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        sparse_lib.launch_sparse_ref_v3(
            A_comp_t.data_ptr(), E_t.data_ptr(), B_t.data_ptr(),
            C_sparse.data_ptr(), M, N, K
        )
    torch.cuda.synchronize()
    sparse_time = (time.time() - t0) / iters
    
    # Metrics
    flops = 2 * M * N * K
    dense_tflops = flops / (dense_time * 1e12)
    sparse_tflops = flops / (sparse_time * 1e12)
    speedup = dense_time / sparse_time
    
    # Correctness
    max_err = (C_sparse - D_ref_t).abs().max().item()
    
    return {
        'dense_ms': dense_time * 1000,
        'sparse_ms': sparse_time * 1000,
        'dense_tflops': dense_tflops,
        'sparse_tflops': sparse_tflops,
        'speedup': speedup,
        'max_err': max_err,
        'correct': max_err < 1e-4
    }

# Test configurations
sizes = [
    (128, 128, 128, "test_128x128x128"),
    (256, 256, 256, "test_256x256x256"),
    (512, 512, 512, "test_512x512x512"),
    (1024, 1024, 1024, "test_1024x1024x1024"),
    (2048, 2048, 2048, "test_2048x2048x2048"),
]

results = []
for M, N, K, prefix in sizes:
    print(f"\n{'='*70}")
    print(f"Testing {M}×{K} @ {K}×{N} = {M}×{N}")
    print(f"{'='*70}")
    
    result = benchmark_size(M, N, K, prefix)
    results.append((M, result))
    
    print(f"Dense (cuBLAS):  {result['dense_ms']:8.4f} ms  |  {result['dense_tflops']:7.2f} TFLOPS")
    print(f"Sparse (v3):     {result['sparse_ms']:8.4f} ms  |  {result['sparse_tflops']:7.2f} TFLOPS")
    print(f"Speedup:         {result['speedup']:7.3f}×")
    print(f"Max error:       {result['max_err']:.2e}  {'✅' if result['correct'] else '❌'}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Size':<12} {'Dense':>10} {'Sparse':>10} {'Speedup':>10} {'Status':>8}")
print(f"{'-'*70}")
for M, r in results:
    status = '✅' if r['correct'] else '❌'
    print(f"{M}×{M:<8} {r['dense_tflops']:>7.2f} TF {r['sparse_tflops']:>7.2f} TF {r['speedup']:>7.2f}× {status:>8}")

print(f"\n💡 Key Observations:")
max_speedup = max(r['speedup'] for _, r in results)
max_tflops = max(r['sparse_tflops'] for _, r in results)
print(f"   Peak speedup: {max_speedup:.2f}×")
print(f"   Peak TFLOPS:  {max_tflops:.2f}")
print(f"   All tests:    {'✅ PASS' if all(r['correct'] for _, r in results) else '❌ FAIL'}")
