"""GPU compatibility detection for SparseFlow"""
import torch
import warnings

def check_sparse_support():
    """
    Check if GPU supports 2:4 sparse Tensor Cores.
    Returns: (supported: bool, message: str)
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    # Get compute capability
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    
    # Ampere (SM80+) required for 2:4 sparse
    if sm >= 80:
        gpu_name = torch.cuda.get_device_name()
        return True, f"✅ {gpu_name} (SM{sm}) supports 2:4 sparse"
    else:
        gpu_name = torch.cuda.get_device_name()
        return False, f"⚠️  {gpu_name} (SM{sm}) does NOT support 2:4 sparse (requires SM80+/Ampere)"

def require_sparse_support():
    """Raise error if sparse not supported"""
    supported, msg = check_sparse_support()
    if not supported:
        raise RuntimeError(
            f"SparseFlow requires NVIDIA Ampere (SM80+) or newer GPUs.\n"
            f"{msg}\n"
            f"Supported: RTX 30-series, RTX 40-series, A100, H100\n"
            f"Not supported: V100, T4, RTX 20-series or older"
        )
    return msg

def warn_if_unsupported():
    """Warn but don't crash if unsupported"""
    supported, msg = check_sparse_support()
    if not supported:
        warnings.warn(
            f"SparseFlow sparse operations not available.\n{msg}\n"
            f"Falling back to dense PyTorch operations.",
            RuntimeWarning
        )
    return supported
