"""Model-level dispatcher: Check policy ONCE, not 154 times"""
import torch
from sparseflow.nn.policy import SparseFlowPolicy

class SparseFlowModel:
    """Wrapper that sets global sparse/dense mode based on batch size"""
    
    def __init__(self, model, policy: SparseFlowPolicy):
        self.model = model
        self.policy = policy
        self._current_mode = {}  # op_name -> use_sparse
    
    def set_mode_for_batch(self, batch_size: int, seq_len: int):
        """Set sparse/dense mode for all layers based on M"""
        M = batch_size * seq_len
        
        # Update all layer modes ONCE
        for name in ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]:
            self._current_mode[name] = self.policy.should_use_sparse(name, M)
        
        # Apply to all layers
        layers = getattr(getattr(self.model, "model", self.model), "layers", [])
        for layer in layers:
            # MLP
            mlp = getattr(layer, "mlp", None)
            if mlp:
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    proj = getattr(mlp, proj_name, None)
                    if proj and hasattr(proj, '_force_mode'):
                        proj._force_mode = self._current_mode.get(proj_name, None)
            
            # Attention
            attn = getattr(layer, "self_attn", None)
            if attn:
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    proj = getattr(attn, proj_name, None)
                    if proj and hasattr(proj, '_force_mode'):
                        proj._force_mode = self._current_mode.get(proj_name, None)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
