"""
SparseFlow - High-performance 2:4 sparse inference for NVIDIA GPUs

Example:
    >>> import sparseflow as sf
    >>> 
    >>> # Check GPU compatibility
    >>> supported, msg = sf.check_sparse_support()
    >>> print(msg)
    >>> 
    >>> # Convert dense layer to sparse
    >>> dense = nn.Linear(4096, 4096)
    >>> sparse = sf.SparseLinear.from_dense(dense, method="magnitude")
    >>> 
    >>> # Use like normal Linear
    >>> x = torch.randn(1, 4096, device='cuda', dtype=torch.float16)
    >>> y = sparse(x)  # 2× faster on Ampere+ GPUs
"""

__version__ = "3.0.0-alpha"

from .core import (
    compress_2_4,
    validate_2_4, 
    prune_2_4,
    sparse_mm,
    check_sparse_support,
    is_sparse_available,
)

from .nn import SparseLinear

__all__ = [
    "compress_2_4",
    "validate_2_4",
    "prune_2_4",
    "sparse_mm",
    "check_sparse_support",
    "is_sparse_available",
    "SparseLinear",
]
