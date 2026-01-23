import torch
import time

def manual_24_prune(dense_tensor):
    """2:4 pruning"""
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

def report_errors(C_ref, C_test, name="Sparse"):
    """Trust-but-verify error reporting"""
    Cr = C_ref.float()
    Ct = C_test.float()

    diff = (Ct - Cr).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rmse = (diff.pow(2).mean().sqrt()).item()

    denom = Cr.abs().clamp_min(1e-8)
    rel = (diff / denom)
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()

    print(f"  C_ref dtype:   {C_ref.dtype}, layout: {C_ref.layout}")
    print(f"  C_{name} dtype: {C_test.dtype}, layout: {C_test.layout}")
    print(f"  Same storage ptr: {C_ref.data_ptr() == C_test.data_ptr()}")
    print(f"  torch.equal:      {torch.equal(C_ref, C_test)}")
    print(f"  Max absolute error: {max_abs:.8e}")
    print(f"  Max relative error: {max_rel:.8e}")
    print(f"  Mean absolute error:{mean_abs:.8e}")
    print(f"  Mean relative error:{mean_rel:.8e}")
    print(f"  RMSE:             {rmse:.8e}")

    # Show worst offenders
    flat = diff.view(-1)
    k = min(5, flat.numel())
    topv, topi = torch.topk(flat, k)
    print("  Top diffs:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"    idx {i}: ref={Cr.view(-1)[i].item():.6f}  test={Ct.view(-1)[i].item():.6f}  diff={v:.6e}")

    return max_abs, max_rel

def validate_correctness(M, N, K, name="Test"):
    print(f"\n{'='*70}")
    print(f"  {name}: M={M}, N={N}, K={K}")
    print(f"{'='*70}")
    
    # Generate random matrices
    torch.manual_seed(42)
    A_original = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # Prune A to 2:4 sparsity
    A_pruned = manual_24_prune(A_original)
    
    # REFERENCE: Dense matmul on the PRUNED matrix
    C_ref = torch.matmul(A_pruned, B).float()
    
    # TEST: Semi-structured (cuSPARSELt / Tensor Cores)
    print("\n--- Semi-structured (2:4 Tensor Cores) ---")
    A_ss = torch.sparse.to_sparse_semi_structured(A_pruned)
    C_ss = torch.matmul(A_ss, B).float()
    max_abs_ss, max_rel_ss = report_errors(C_ref, C_ss, name="semi_structured")
    
    # SANITY CHECK: Corrupted version (must FAIL)
    print("\n--- Sanity check (should FAIL) ---")
    C_bad = C_ss.clone()
    C_bad.view(-1)[0] += 1.0
    report_errors(C_ref, C_bad, name="corrupted")
    
    # Pass/fail
    passed = max_abs_ss < 1e-3 and max_rel_ss < 0.01
    print(f"\n  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return passed

print("=" * 70)
print("  SparseFlow REAL Correctness Validation")
print("  (Using Semi-Structured Sparse Tensor Cores)")
print("=" * 70)

tests = [
    (512, 512, 512, "Small square"),
    (1024, 1024, 1024, "Medium square"),
    (2048, 2048, 2048, "Large square"),
    (128, 4096, 4096, "LLaMA attn (seq=128)"),
    (512, 4096, 4096, "LLaMA attn (seq=512)"),
]

passed = sum(validate_correctness(M, N, K, name) for M, N, K, name in tests)
total = len(tests)

print("\n" + "=" * 70)
print(f"  Summary: {passed}/{total} tests passed")
print("=" * 70)

if passed == total:
    print("\n✅ ALL TESTS PASSED - Sparse Tensor Cores work correctly!")
else:
    print(f"\n❌ {total - passed} TESTS FAILED - Kernel has bugs")
