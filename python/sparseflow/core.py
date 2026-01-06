"""Core SparseFlow operations"""

import torch
import warnings

# Try to import C++ extension
try:
    from . import sparseflow_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    warnings.warn(
        "SparseFlow C++ extension not available. "
        "Using fallback implementations (slower).",
        RuntimeWarning
    )

def check_sparse_support():
    """
    Check if current GPU supports 2:4 sparse operations.
    
    Returns:
        tuple: (supported: bool, message: str)
    """
    if not _CPP_AVAILABLE:
        return False, "C++ extension not available"
    
    return sparseflow_cpp.check_sparse_support()

def is_sparse_available():
    """Check if 2:4 sparse is available on current GPU"""
    if not _CPP_AVAILABLE:
        return False
    
    supported, _ = check_sparse_support()
    return supported

def validate_2_4(tensor):
    """
    Validate tensor has valid 2:4 sparsity pattern.
    
    Args:
        tensor: PyTorch tensor to validate
        
    Returns:
        bool: True if valid 2:4 pattern
    """
    if not _CPP_AVAILABLE:
        raise RuntimeError("C++ extension required for validation")
    
    return sparseflow_cpp.validate_2_4(tensor)

def prune_2_4(tensor, method="magnitude"):
    """
    Prune dense tensor to 2:4 sparsity pattern.
    
    Args:
        tensor: Dense input tensor
        method: Pruning method ("magnitude", "random")
        
    Returns:
        Pruned tensor with 2:4 pattern
    """
    if not _CPP_AVAILABLE:
        raise RuntimeError("C++ extension required for pruning")
    
    if method == "magnitude":
        return sparseflow_cpp.prune_2_4_magnitude(tensor)
    elif method == "random":
        # TODO: Implement random pruning
        raise NotImplementedError("Random pruning not yet implemented")
    else:
        raise ValueError(f"Unknown pruning method: {method}")

def compress_2_4(tensor):
    """
    Compress tensor with 2:4 pattern to compressed format.
    
    Args:
        tensor: Tensor with valid 2:4 pattern
        
    Returns:
        Compressed tensor (50% size)
    """
    if not _CPP_AVAILABLE:
        raise RuntimeError("C++ extension required for compression")
    
    # Validate pattern
    if not validate_2_4(tensor):
        raise ValueError("Tensor does not have valid 2:4 sparsity pattern")
    
    return sparseflow_cpp.compress_2_4(tensor)

def sparse_mm(A, Bc, epilogue="none", bias=None):
    """
    Sparse matrix multiply: C = A @ Bc
    
    Args:
        A: Dense matrix (M×K, FP16)
        Bc: Compressed sparse matrix (K×N, FP16)
        epilogue: Activation function ("none", "relu", "silu", "gelu")
        bias: Optional bias vector (N elements)
        
    Returns:
        Result matrix (M×N, FP16)
    """
    if not _CPP_AVAILABLE:
        raise RuntimeError("C++ extension required for sparse operations")
    
    if not is_sparse_available():
        raise RuntimeError(
            "2:4 sparse not available on this GPU. "
            "Requires NVIDIA Ampere (SM80+) or newer."
        )
    
    return sparseflow_cpp.sparse_mm_fused(A, Bc, epilogue, bias)
