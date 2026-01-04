import torch
from .compat import warn_if_unsupported

# Check on import
_SPARSE_AVAILABLE = warn_if_unsupported()

def compress_2_4(B: torch.Tensor) -> torch.Tensor:
    """Compress matrix using NVIDIA 2:4 structured sparsity"""
    if not _SPARSE_AVAILABLE:
        raise RuntimeError(
            "2:4 sparse not available on this GPU. "
            "Requires NVIDIA Ampere (SM80+) or newer."
        )
    
    assert B.is_cuda and B.dtype == torch.float16
    return torch._cslt_compress(B)

def sparse_mm(A: torch.Tensor, Bc: torch.Tensor) -> torch.Tensor:
    """SparseFlow sparse GEMM: C = A @ Bc"""
    if not _SPARSE_AVAILABLE:
        raise RuntimeError(
            "2:4 sparse not available on this GPU. "
            "Requires NVIDIA Ampere (SM80+) or newer."
        )
    
    assert A.is_cuda and Bc.is_cuda
    return torch._cslt_sparse_mm(Bc, A.T).T

def is_sparse_available() -> bool:
    """Check if 2:4 sparse is available on current GPU"""
    return _SPARSE_AVAILABLE
