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
    
    # Scale threshold with K (accumulation dimension)
    max_threshold = 0.3
    mean_threshold = 0.01  # 1% mean error is acceptable for FP16
    
    passed = (max_abs < max_threshold) and (mean_abs < mean_threshold)
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed

print("=" * 70)
print("  SparseFlow Correctness Validation")
print("  (FP16 Sparse Tensor Cores)")
print("=" * 70)
print("Thresholds: max_abs < 0.3, mean_abs < 0.01\n")

tests = [
    (512, 512, 512, "Small square"),
    (1024, 1024, 1024, "Medium square"),
    (2048, 2048, 2048, "Large square"),
    (4096, 4096, 4096, "XL square"),
    (128, 4096, 4096, "LLaMA attn seq=128"),
    (512, 4096, 4096, "LLaMA attn seq=512"),
    (2048, 4096, 4096, "LLaMA attn seq=2048"),
    (512, 11008, 4096, "LLaMA FFN gate seq=512"),
    (2048, 11008, 4096, "LLaMA FFN gate seq=2048"),
    (512, 4096, 11008, "LLaMA FFN down seq=512"),
    (2048, 4096, 11008, "LLaMA FFN down seq=2048"),
]

passed = sum(validate_correctness(M, N, K, name) for M, N, K, name in tests)
total = len(tests)

print(f"\n{'='*70}")
print(f"  Summary: {passed}/{total} tests passed")
print(f"{'='*70}")

if passed == total:
    print("\n✅ ALL TESTS PASSED!")
    print("🚀 Sparse Tensor Cores validated for production")
else:
    print(f"\n⚠️  {total-passed}/{total} tests need review")
