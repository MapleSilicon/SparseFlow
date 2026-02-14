"""SparseFlowMLP: Optimized MLP replacement"""
import torch
import torch.nn as nn
from typing import Optional
from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.nn.sparseflow_linear import prune_24_dense_weight

class SparseFlowMLP(nn.Module):
    """Drop-in replacement for LlamaMLP using sparse tensor cores."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_bias: Optional[torch.Tensor] = None,
        up_bias: Optional[torch.Tensor] = None,
        down_bias: Optional[torch.Tensor] = None,
        policy: SparseFlowPolicy = SparseFlowPolicy(),
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.policy = policy
        
        # Convert to sparse - store as SparseSemiStructuredTensor
        gate_pruned = prune_24_dense_weight(gate_weight.contiguous())
        self.register_buffer("gate_sparse", 
                           torch.sparse.to_sparse_semi_structured(gate_pruned), 
                           persistent=False)
        self.register_buffer("gate_bias", gate_bias.contiguous() if gate_bias is not None else None)
        
        up_pruned = prune_24_dense_weight(up_weight.contiguous())
        self.register_buffer("up_sparse",
                           torch.sparse.to_sparse_semi_structured(up_pruned),
                           persistent=False)
        self.register_buffer("up_bias", up_bias.contiguous() if up_bias is not None else None)
        
        down_pruned = prune_24_dense_weight(down_weight.contiguous())
        self.register_buffer("down_sparse",
                           torch.sparse.to_sparse_semi_structured(down_pruned),
                           persistent=False)
        self.register_buffer("down_bias", down_bias.contiguous() if down_bias is not None else None)

    def _spmm_pick(self, Ws, xT):
        """Auto-pick packed/meta orientation + handle alignment (K must be multiple of 16)"""
        # Pad xT's last dimension (N/tokens) to multiple of 16 for CUTLASS alignment
        orig_n = xT.shape[1]
        pad_val = (16 - (orig_n % 16)) % 16
        
        if pad_val > 0:
            xT = torch.nn.functional.pad(xT, (0, pad_val))
        
        # Pick orientation based on shape
        if Ws.data.shape[1] == xT.shape[0]:
            res = torch.ops.aten._sparse_semi_structured_mm(Ws.packed, Ws.meta, xT)
        elif Ws.data.shape[0] == xT.shape[0]:
            res = torch.ops.aten._sparse_semi_structured_mm(Ws.packed_t, Ws.meta_t, xT)
        else:
            raise RuntimeError(
                f"Shape mismatch: Ws.data={tuple(Ws.data.shape)} xT={tuple(xT.shape)}"
            )
        
        # Slice back to original size
        if pad_val > 0:
            res = res[:, :orig_n]
        
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using low-level sparse ops"""
        # Reshape to 2D
        leading = x.shape[:-1]
        T = 1
        for d in leading:
            T *= int(d)
        
        x_2d = x.reshape(T, self.hidden_size).contiguous()  # [T, H]
        
        # Gate projection - use transpose approach that we know works
        xT = x_2d.transpose(0, 1)  # [H, T].contiguous()
        gateT = self._spmm_pick(self.gate_sparse, xT)  # [I, T]
        gate = gateT.transpose(0, 1)  # [T, I]
        if self.gate_bias is not None:
            gate = gate + self.gate_bias
        
        # Up projection
        upT = self._spmm_pick(self.up_sparse, xT)  # [I, T]
        up = upT.transpose(0, 1)  # [T, I]
        if self.up_bias is not None:
            up = up + self.up_bias
        
        # Activation
        hidden = torch.nn.functional.silu(gate) * up  # [T, I]
        
        # Down projection
        hT = hidden.transpose(0, 1)  # [I, T]
        outT = self._spmm_pick(self.down_sparse, hT)  # [H, T]
        out = outT.transpose(0, 1)  # [T, H]
        if self.down_bias is not None:
            out = out + self.down_bias
        
        return out.reshape(*leading, self.hidden_size)


def make_sparseflow_mlp(mlp_module, policy: SparseFlowPolicy = SparseFlowPolicy()):
    """Convert LlamaMLP to SparseFlowMLP"""
    return SparseFlowMLP(
        hidden_size=mlp_module.gate_proj.in_features,
        intermediate_size=mlp_module.gate_proj.out_features,
        gate_weight=mlp_module.gate_proj.weight.data,
        up_weight=mlp_module.up_proj.weight.data,
        down_weight=mlp_module.down_proj.weight.data,
        policy=policy,
    )
