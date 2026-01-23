#!/usr/bin/env python3
"""
SparseFlow Production Validation
Validates sparse tensor cores produce numerically acceptable results
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = False

def manual_24_prune(dense_tensor):
    M, K = dense_tensor.shape
    pruned = torch.zeros_like(dense_tensor)
    for i in range(M):
        for j in range(0, K, 4):
            block = dense_tensor[i, j:j+4]
            _, indices = torch.topk(torch.abs(block), k=2, sorted=False)
            for idx in indices:
                pruned[i, j + idx] = block[idx]
    return pruned

def validate(M, N, K, name):
    print(f"\n{name}: M={M}, N={N}, K={K}")
    
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    
    A_pruned = manual_24_prune(A)
    C_ref_fp32 = (A_pruned.float() @ B.float())
    
    A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)
    C_sparse = (A_sparse @ B).float()
    
    # Compare against FP32 truth (not dense FP16!)
    err = torch.abs(C_ref_fp32 - C_sparse)
    max_err = err.max().item()
    mean_err = err.mean().item()
    
    print(f"  Max error: {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6e}")
    
    # FP16 output quantization threshold: max < 0.2 is excellent
    passed = max_err < 0.2
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    return passed

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

print("="*70)
print("SparseFlow Production Validation")
print("="*70)
print("Comparing sparse FP16 output against FP32 ground truth")
print("Threshold: max_error < 0.2 (FP16 quantization appropriate)\n")

passed = sum(validate(M, N, K, n) for M, N, K, n in tests)
print(f"\n{'='*70}")
print(f"Summary: {passed}/{len(tests)} passed")
print(f"{'='*70}")

if passed == len(tests):
    print("\n✅ ALL TESTS PASSED!")
    print("🚀 SparseFlow is production-ready")
    exit(0)
else:
    print(f"\n❌ {len(tests)-passed} tests failed")
    exit(1)
