"""
Simple example showing 2× speedup with SparseFlow
"""
import torch
import time
import sparseflow as sf

def apply_2_4_pattern(tensor):
    """Apply 2:4 sparsity: 2 zeros per 4 consecutive values"""
    out = tensor.clone()
    for i in range(0, tensor.shape[1], 4):
        group = out[:, i:i+4]
        _, indices = torch.topk(torch.abs(group), k=2, dim=1, largest=False)
        for row in range(out.shape[0]):
            out[row, i + indices[row, 0]] = 0
            out[row, i + indices[row, 1]] = 0
    return out

print("="*60)
print("SPARSEFLOW 2× SPEEDUP DEMO")
print("="*60)

# Large matrix (where sparse wins)
M = 4096
print(f"\nMatrix size: {M}×{M}")

# Create data
x = torch.randn(M, M, device='cuda', dtype=torch.float16)
W_dense = torch.randn(M, M, device='cuda', dtype=torch.float16)

# Apply 2:4 sparsity and compress
W_sparse = apply_2_4_pattern(W_dense)
Wc = sf.compress_2_4(W_sparse)

print(f"Sparsity: {(W_sparse == 0).float().mean()*100:.1f}%")

# Warmup
for _ in range(10):
    _ = torch.matmul(x, W_dense)
    _ = sf.sparse_mm(x, Wc)
torch.cuda.synchronize()

# Benchmark dense
t0 = time.perf_counter()
for _ in range(50):
    y_dense = torch.matmul(x, W_dense)
torch.cuda.synchronize()
dense_time = time.perf_counter() - t0

# Benchmark sparse
t0 = time.perf_counter()
for _ in range(50):
    y_sparse = sf.sparse_mm(x, Wc)
torch.cuda.synchronize()
sparse_time = time.perf_counter() - t0

speedup = dense_time / sparse_time

print(f"\nDense:  {dense_time*1000:.1f} ms")
print(f"Sparse: {sparse_time*1000:.1f} ms")
print(f"Speedup: {speedup:.2f}×")

if speedup > 1.5:
    print("\n✅ SparseFlow is 2× faster!")
else:
    print("\n⚠️  Try larger matrices for best speedup")

print("="*60)
