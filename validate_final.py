import torch

def manual_24_prune(dense_tensor):
    M, K = dense_tensor.shape
    pruned = torch.zeros_like(dense_tensor)
    for i in range(M):
        for j in range(0, K, 4):
            block = dense_tensor[i, j:j+4]
            abs_vals = torch.abs(block)
            _, indices = torch.topk(abs_vals, k=2)
            for idx in indices:
                pruned[i, j + idx] = block[idx]
    return pruned

def validate_correctness(M, N, K, name="Test"):
    print(f"\n{'='*70}")
    print(f"  {name}: M={M}, N={N}, K={K}")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    A_original = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    A_pruned = manual_24_prune(A_original)
    C_ref = torch.matmul(A_pruned, B).float()
    
    A_ss = torch.sparse.to_sparse_semi_structured(A_pruned)
    C_ss = torch.matmul(A_ss, B).float()
    
    diff = (C_ss - C_ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    
    print(f"  Max absolute error: {max_abs:.6f}")
    print(f"  Mean absolute error: {mean_abs:.6e}")
    
    passed = (max_abs < 0.3) and (mean_abs < 1e-3)
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed

print("=" * 70)
print("  SparseFlow Sparse Tensor Core Validation")
print("=" * 70)

tests = [
    (512, 512, 512, "Small"),
    (1024, 1024, 1024, "Medium"),
    (2048, 2048, 2048, "Large"),
    (128, 4096, 4096, "LLaMA attn 128"),
    (512, 4096, 4096, "LLaMA attn 512"),
]

passed = sum(validate_correctness(M, N, K, name) for M, N, K, name in tests)
print(f"\nSummary: {passed}/{len(tests)} passed")
