"""Dual-compiled models: one dense, one sparse (no runtime branching)"""
import torch
from typing import Optional

class DualCompiledModel:
    """Wrapper that compiles dense + sparse separately, dispatches based on M"""
    
    def __init__(self, model_dense, model_sparse, policy, threshold_M: int = 8192):
        """
        Args:
            model_dense: Model with SparseFlow layers forced to dense
            model_sparse: Model with SparseFlow layers forced to sparse
            policy: SparseFlowPolicy (for future runtime decisions)
            threshold_M: Switch to dense above this M (to avoid overhead)
        """
        self.model_dense = model_dense
        self.model_sparse = model_sparse
        self.threshold_M = threshold_M
        self._compiled_dense = None
        self._compiled_sparse = None
    
    def compile(self, mode: str = "max-autotune"):
        """Compile both graphs"""
        print(f"Compiling dense graph (mode={mode})...")
        self._compiled_dense = torch.compile(self.model_dense, mode=mode)
        
        print(f"Compiling sparse graph (mode={mode})...")
        self._compiled_sparse = torch.compile(self.model_sparse, mode=mode)
        
        return self
    
    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        """Dispatch to dense or sparse based on batch size"""
        # Calculate M from input shape
        if input_ids is not None:
            M = input_ids.shape[0] * input_ids.shape[1]  # batch * seq
        else:
            # Fallback: use sparse for smaller batches
            M = 1024
        
        # Dispatch OUTSIDE compiled graph (no branching inside)
        if M <= self.threshold_M:
            return self._compiled_sparse(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        else:
            return self._compiled_dense(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

def force_sparse_mode(model, force_value: bool):
    """Force all SparseFlowLinear layers to use sparse or dense"""
    from sparseflow.nn.sparseflow_linear import SparseFlowLinear
    
    count = 0
    for module in model.modules():
        if isinstance(module, SparseFlowLinear):
            # Override should_use_sparse to always return force_value
            module._force_mode = force_value
            count += 1
    
    return count
