"""SparseFlow - High-performance 2:4 sparse inference for NVIDIA GPUs"""
from .ops import compress_2_4, sparse_mm

__version__ = "2.0.0"
__all__ = ["compress_2_4", "sparse_mm"]
