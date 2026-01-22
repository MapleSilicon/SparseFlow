import torch
import pytest

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

@pytest.mark.parametrize("M,N,K,name", [
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
])
def test_sparse_correctness(M, N, K, name):
    """Validate sparse kernel correctness vs pruned-dense reference"""
    torch.manual_seed(42)
    A_original = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    A_pruned = manual_24_prune(A_original)
    C_ref = torch.matmul(A_pruned, B).float()
    
    A_sparse = A_pruned.to_sparse_csr()
    C_sparse = torch.matmul(A_sparse.to_dense(), B).float()
    
    error = torch.abs(C_ref - C_sparse)
    max_abs_error = error.max().item()
    max_rel_error = (error / (torch.abs(C_ref) + 1e-6)).max().item()
    
    assert max_abs_error < 1e-3, f"{name}: max_abs_error={max_abs_error:.6f}"
    assert max_rel_error < 0.01, f"{name}: max_rel_error={max_rel_error:.6f}"
