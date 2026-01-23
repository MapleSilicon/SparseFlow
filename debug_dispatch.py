import torch
import numpy as np

def manual_24_prune(dense_tensor):
    M, K = dense_tensor.shape
    pruned = torch.zeros_like(dense_tensor)
    for i in range(M):
        for j in range(0, K, 4):
            block = dense_tensor[i, j:j+4]
            abs_vals = torch.abs(block)
            _, indices = torch.topk(abs_vals, k=2, sorted=False)
            for idx in indices:
                pruned[i, j + idx] = block[idx]
    return pruned

print("="*70)
print("CHECK 1: Determinism - Run 10 times")
print("="*70)

errors = []
for run in range(10):
    torch.manual_seed(42)
    A = torch.randn(512, 4096, dtype=torch.float16, device='cuda')
    B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
    
    A_pruned = manual_24_prune(A)
    C_ref = torch.matmul(A_pruned, B).float()
    
    A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)
    C_sparse = torch.matmul(A_sparse, B).float()
    
    err = torch.abs(C_ref - C_sparse).max().item()
    errors.append(err)
    print(f"Run {run+1}: max_error = {err:.6f}")

print(f"\nError variance: {np.std(errors):.8f}")
if np.std(errors) > 1e-6:
    print("❌ NON-DETERMINISTIC - Race condition or uninitialized memory")
else:
    print("✅ DETERMINISTIC - Math/config bug")

print("\n" + "="*70)
print("CHECK 2: Locate max error position")
print("="*70)

torch.manual_seed(42)
A = torch.randn(512, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

A_pruned = manual_24_prune(A)
C_ref = torch.matmul(A_pruned, B).float()

A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)
C_sparse = torch.matmul(A_sparse, B).float()

diff = torch.abs(C_ref - C_sparse)
max_idx = diff.argmax().item()
i, j = max_idx // 4096, max_idx % 4096

print(f"Max error at: ({i}, {j})")
print(f"Reference value: {C_ref[i, j].item():.6f}")
print(f"Sparse value: {C_sparse[i, j].item():.6f}")
print(f"Error: {diff[i, j].item():.6f}")

# Show 3x3 neighborhood
print(f"\nNeighborhood around ({i}, {j}):")
for di in range(-1, 2):
    for dj in range(-1, 2):
        ni, nj = i+di, j+dj
        if 0 <= ni < 512 and 0 <= nj < 4096:
            print(f"  ({ni},{nj}): ref={C_ref[ni,nj].item():.4f} sparse={C_sparse[ni,nj].item():.4f} diff={diff[ni,nj].item():.4f}")

print("\n" + "="*70)
print("CHECK 3: Error distribution (tiling pattern?)")
print("="*70)

# Reshape to 8x8 tiles and check tile-level errors
tiles_i = diff.view(512//8, 8, 4096//8, 8).permute(0, 2, 1, 3).reshape(-1, 64)
tile_max_errors = tiles_i.max(dim=1)[0]
worst_tiles = torch.topk(tile_max_errors, 5)

print(f"Top 5 worst tiles (out of {len(tile_max_errors)}):")
for idx, err in zip(worst_tiles.indices.tolist(), worst_tiles.values.tolist()):
    tile_i = idx // (4096//8)
    tile_j = idx % (4096//8)
    print(f"  Tile ({tile_i}, {tile_j}): max_error = {err:.6f}")

print("\n" + "="*70)
print("CHECK 4: Compare M=128 (PASS) vs M=512 (FAIL)")
print("="*70)

for M in [128, 512]:
    torch.manual_seed(42)
    A = torch.randn(M, 4096, dtype=torch.float16, device='cuda')
    B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
    
    A_pruned = manual_24_prune(A)
    C_ref = torch.matmul(A_pruned, B).float()
    
    A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)
    C_sparse = torch.matmul(A_sparse, B).float()
    
    diff = torch.abs(C_ref - C_sparse)
    print(f"M={M}: max_err={diff.max().item():.6f}, mean_err={diff.mean().item():.6e}")

print("\n" + "="*70)
print("CHECK 5: Explicit zero-init test")
print("="*70)

torch.manual_seed(42)
A = torch.randn(512, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

A_pruned = manual_24_prune(A)
C_ref = torch.matmul(A_pruned, B).float()

# Force zero output
C_out = torch.zeros(512, 4096, dtype=torch.float32, device='cuda')
A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)
C_sparse = torch.matmul(A_sparse, B).float()

diff = torch.abs(C_ref - C_sparse)
print(f"With explicit zero: max_err={diff.max().item():.6f}, mean_err={diff.mean().item():.6e}")
