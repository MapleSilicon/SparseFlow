"""PyTorch nn.Module wrappers for SparseFlow"""

import torch
from torch import nn
from . import core as sf

class SparseLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with 2:4 sparsity.
    
    IMPORTANT: Weights must already be in 2:4 pattern.
    Use SparseLinear.from_dense() for automatic conversion.
    
    Args:
        weight_compressed: Compressed weight matrix
        metadata: Sparsity metadata (currently unused)
        bias: Optional bias vector
        
    Example:
        >>> # Convert existing layer
        >>> dense = nn.Linear(4096, 4096)
        >>> sparse = SparseLinear.from_dense(dense)
        >>> 
        >>> # Use like normal Linear
        >>> x = torch.randn(1, 4096, device='cuda', dtype=torch.float16)
        >>> y = sparse(x)  # 2× faster on Ampere+ GPUs
    """
    
    def __init__(self, weight_compressed, metadata=None, bias=None):
        super().__init__()
        self.weight_compressed = nn.Parameter(weight_compressed)
        self.metadata = metadata  # For future use
        
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)
    
    @staticmethod
    def from_dense(
        dense_linear,
        method="magnitude",
        validate=True,
        return_diff=False
    ):
        """
        Convert dense Linear to sparse.
        
        Args:
            dense_linear: Original nn.Linear module
            method: Pruning method ("magnitude", "random")
            validate: Check 2:4 pattern compliance
            return_diff: Return accuracy impact metrics
            
        Returns:
            sparse_linear: SparseLinear module
            diff_report: Optional accuracy report (if return_diff=True)
        """
        # Get weight
        weight = dense_linear.weight.data
        
        # Prune to 2:4 pattern
        weight_sparse = sf.prune_2_4(weight, method=method)
        
        # Validate
        if validate:
            if not sf.validate_2_4(weight_sparse):
                raise ValueError("Failed to create valid 2:4 pattern")
        
        # Compress
        weight_compressed = sf.compress_2_4(weight_sparse)
        
        # Create sparse linear
        sparse_linear = SparseLinear(
            weight_compressed,
            metadata=None,
            bias=dense_linear.bias.data if dense_linear.bias is not None else None
        )
        
        # Measure accuracy impact if requested
        if return_diff:
            with torch.no_grad():
                # Test input
                in_features = dense_linear.in_features
                test_input = torch.randn(
                    128, in_features,
                    device=weight.device,
                    dtype=weight.dtype
                )
                
                # Dense output
                dense_out = dense_linear(test_input)
                
                # Sparse output
                sparse_out = sparse_linear(test_input)
                
                # Compute differences
                diff = {
                    'max_error': (dense_out - sparse_out).abs().max().item(),
                    'mean_error': (dense_out - sparse_out).abs().mean().item(),
                    'relative_error': (
                        (dense_out - sparse_out).abs() / 
                        (dense_out.abs() + 1e-8)
                    ).mean().item(),
                }
            
            return sparse_linear, diff
        
        return sparse_linear
    
    def forward(self, x):
        """
        Forward pass through sparse linear layer.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        # Reshape input to 2D for matmul
        orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        
        # Sparse matmul
        output = sf.sparse_mm(
            x_2d, 
            self.weight_compressed.t(),
            epilogue="none"
        )
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        # Reshape back
        output = output.view(*orig_shape[:-1], -1)
        
        return output
    
    def extra_repr(self):
        return f'in_features={self.weight_compressed.shape[1]}, out_features={self.weight_compressed.shape[0]}, bias={self.bias is not None}'
