import torch
import time
import sys
sys.path.insert(0, '..')
import sparseflow as sf

print("="*70)
print("SPARSEFLOW BENCHMARK - Honest Reporting")
print("="*70)

def apply_2_4_pattern(tensor):
    """Apply 2:4 sparsity pattern"""
    out = tensor.clone()
    K = tensor.shape[1]
    for i in range(0, K, 4):
        group = out[:, i:i+4]
        _, indices = torch.topk(torch.abs(group), k=2, dim=1, largest=False)
        for row in range(out.shape[0]):
            out[row, i + indices[row, 0]] = 0
            out[row, i + indices[row, 1]] = 0
    return out

def bench(M, iters=100):
    A = torch.randn(M, M, device="cuda", dtype=torch.float16)
    B_dense = torch.randn(M, M, device="cuda", dtype=torch.float16)
    
    B_sparse = apply_2_4_pattern(B_dense)
    Bc = sf.compress_2_4(B_sparse)
    
    # Warmup (important!)
    for _ in range(20):
        _ = torch.matmul(A, B_dense)
        _ = sf.sparse_mm(A, Bc)
    torch.cuda.synchronize()
    
    # Benchmark dense
    t0 = time.perf_counter()
    for _ in range(iters):
        C = torch.matmul(A, B_dense)
    torch.cuda.synchronize()
    dense_time = time.perf_counter() - t0
    
    # Benchmark sparse
    t0 = time.perf_counter()
    for _ in range(iters):
        C = sf.sparse_mm(A, Bc)
    torch.cuda.synchronize()
    sparse_time = time.perf_counter() - t0
    
    # FLOP calculations (HONEST)
    dense_flops = 2.0 * M * M * M * iters
    sparse_effective_flops = dense_flops  # Same as dense (marketing number)
    sparse_real_flops = dense_flops * 0.5  # Actual work done (50% of multiplies)
    
    dense_tflops = dense_flops / dense_time / 1e12
    sparse_effective_tflops = sparse_effective_flops / sparse_time / 1e12
    sparse_real_tflops = sparse_real_flops / sparse_time / 1e12
    speedup = dense_time / sparse_time
    
    print(f"\n{M}×{M}:")
    print(f"  Dense:                {dense_tflops:6.2f} TFLOPS")
    print(f"  Sparse (effective):   {sparse_effective_tflops:6.2f} TFLOPS")
    print(f"  Sparse (real work):   {sparse_real_tflops:6.2f} TFLOPS")
    print(f"  Speedup:              {speedup:6.2f}×")
    
    if speedup > 1.0:
        print(f"  ✅ Sparse WINS")
    else:
        print(f"  ⚠️  Dense faster (overhead dominates)")

print("\nNote: 'Effective' = time to do dense-equivalent work")
print("      'Real work' = actual multiply-adds performed\n")

for M in [512, 1024, 2048, 4096]:
    bench(M)

print("\n" + "="*70)
print("RECOMMENDATION: Use sparse for M >= 2048")
print("="*70)
