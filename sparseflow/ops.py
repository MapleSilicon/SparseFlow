import torch

def compress_2_4(B: torch.Tensor) -> torch.Tensor:
    """Compress matrix using NVIDIA 2:4 structured sparsity"""
    assert B.is_cuda and B.dtype == torch.float16
    return torch._cslt_compress(B)

def sparse_mm(A: torch.Tensor, Bc: torch.Tensor) -> torch.Tensor:
    """SparseFlow sparse GEMM: C = A @ Bc"""
    assert A.is_cuda and Bc.is_cuda
    return torch._cslt_sparse_mm(Bc, A.T).T
