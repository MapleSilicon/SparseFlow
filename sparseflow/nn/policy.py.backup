"""Per-operation autotuned policy for sparse vs dense decision"""
from dataclasses import dataclass

@dataclass
class SparseFlowPolicy:
    """
    Per-op policy based on A100 microbenchmark data.
    
    Autotuned thresholds:
    - gate/up: 1.33× at M=256 → use sparse when M >= 256
    - down:    0.98× at M=256, 1.20× at M=512 → use sparse when M >= 512
    """
    min_M_default: int = 512
    
    # MLP thresholds (autotuned from benchmarks)
    min_M_gate_up: int = 256   # Wins from M=256+
    min_M_down: int = 512       # Needs M=512+ to win
    
    # Attention thresholds (conservative until tuned)
    min_M_qkv: int = 512
    min_M_o: int = 512
    
    @staticmethod
    def supports_sm80() -> bool:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    
    def should_use_sparse(self, op_name: str, M: int) -> bool:
        """Decide if sparse should be used for this op and batch size"""
        if M <= 0:
            return False
        
        # Normalize op name
        op_lower = op_name.lower()
        
        # MLP ops
        if "gate" in op_lower or "up" in op_lower:
            return M >= self.min_M_gate_up
        if "down" in op_lower:
            return M >= self.min_M_down
        
        # Attention ops
        if any(x in op_lower for x in ["q_proj", "k_proj", "v_proj"]):
            return M >= self.min_M_qkv
        if "o_proj" in op_lower or "o" == op_lower:
            return M >= self.min_M_o
        
        # Default fallback
        return M >= self.min_M_default
