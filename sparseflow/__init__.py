"""SparseFlow - High-performance 2:4 sparse inference for NVIDIA GPUs"""
from .ops import compress_2_4, sparse_mm, is_sparse_available
from .compat import check_sparse_support

__version__ = "2.1.0"
__all__ = ["compress_2_4", "sparse_mm", "is_sparse_available", "check_sparse_support"]
