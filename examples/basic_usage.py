import torch
import sys
sys.path.insert(0, '..')
import sparseflow as sf

print("SparseFlow Basic Usage Example")
print("="*60)

# Create matrices
M = 1024
A = torch.randn(M, M, device="cuda", dtype=torch.float16)
B = torch.randn(M, M, device="cuda", dtype=torch.float16)

# Apply 2:4 pattern
B_sparse = B.clone()
for i in range(0, M, 4):
    group = B_sparse[:, i:i+4]
    _, indices = torch.topk(torch.abs(group), k=2, dim=1, largest=False)
    for row in range(M):
        B_sparse[row, i + indices[row, 0]] = 0
        B_sparse[row, i + indices[row, 1]] = 0

# Compress and run
Bc = sf.compress_2_4(B_sparse)
C = sf.sparse_mm(A, Bc)

print(f"Input A: {A.shape}")
print(f"Input B (compressed): {Bc.shape}")
print(f"Output C: {C.shape}")
print(f"Sparsity: {(B_sparse == 0).float().mean()*100:.1f}%")
print("✅ Success!")
