import torch

# Check if PyTorch has sparse tensor core support
print("Checking PyTorch sparse support...")

# Try the _sparse_semi_structured_linear function (NVIDIA's 2:4 sparse)
if hasattr(torch, '_sparse_semi_structured_linear'):
    print("✅ PyTorch has sparse tensor core support!")
    
    # Create a simple test
    M, K, N = 16, 32, 8
    A_dense = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    
    # Apply 2:4 sparsity pattern to A
    A_sparse = A_dense.clone()
    for i in range(0, K, 4):
        group = A_sparse[:, i:i+4].abs()
        _, indices = torch.topk(group, k=2, dim=1, largest=False)
        for row in range(M):
            A_sparse[row, i + indices[row, 0]] = 0
            A_sparse[row, i + indices[row, 1]] = 0
    
    print(f"Created {M}×{K} sparse matrix")
    print(f"Sparsity: {(A_sparse == 0).float().mean()*100:.1f}%")
    
else:
    print("❌ PyTorch doesn't have sparse_semi_structured_linear")
    print("Will need to implement from scratch")

# Check what's available
print("\nAvailable sparse functions:")
sparse_funcs = [attr for attr in dir(torch) if 'sparse' in attr.lower()]
for func in sparse_funcs[:10]:
    print(f"  - {func}")
