"""Basic SparseFlow usage example"""

import torch
from torch import nn
import sparseflow as sf

# Check GPU support
print("="*60)
print("GPU COMPATIBILITY CHECK")
print("="*60)
supported, msg = sf.check_sparse_support()
print(msg)

if not supported:
    print("\n⚠️  Sparse operations not available on this GPU")
    print("Exiting...")
    exit(0)

print("\n" + "="*60)
print("CONVERTING DENSE LAYER TO SPARSE")
print("="*60)

# Create dense layer
dense = nn.Linear(4096, 4096).cuda().half()
print(f"Dense layer: {dense.weight.shape}")
print(f"Dense parameters: {dense.weight.numel():,}")

# Convert to sparse (with accuracy report)
sparse, diff = sf.SparseLinear.from_dense(
    dense,
    method="magnitude",
    return_diff=True
)

print(f"\nSparse layer: {sparse.weight_compressed.shape}")
print(f"Sparse parameters: {sparse.weight_compressed.numel():,}")
print(f"Compression: {(1 - sparse.weight_compressed.numel() / dense.weight.numel()) * 100:.1f}%")

print(f"\nAccuracy impact:")
print(f"  Max error: {diff['max_error']:.6f}")
print(f"  Mean error: {diff['mean_error']:.6f}")
print(f"  Relative error: {diff['relative_error']:.6f}")

print("\n" + "="*60)
print("INFERENCE SPEED COMPARISON")
print("="*60)

# Benchmark
x = torch.randn(128, 4096, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(10):
    _ = dense(x)
    _ = sparse(x)

# Dense
torch.cuda.synchronize()
import time
start = time.time()
for _ in range(100):
    _ = dense(x)
torch.cuda.synchronize()
dense_time = time.time() - start

# Sparse
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = sparse(x)
torch.cuda.synchronize()
sparse_time = time.time() - start

print(f"Dense time: {dense_time*1000:.2f} ms")
print(f"Sparse time: {sparse_time*1000:.2f} ms")
print(f"Speedup: {dense_time/sparse_time:.2f}×")

print("\n✅ Example complete!")
