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

    # Maximum M thresholds (efficiency frontier)
    max_M_gate_up: int = 2048
    max_M_down: int = 2048
    max_M_qkv: int = 2048
    max_M_o: int = 2048
    
    @staticmethod
    def supports_sm80() -> bool:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    
    def load_runtime_policy(self, path: str):
        """Load min_m and max_m from JSON policy file"""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        
        ops = data.get("ops", data)
        for op, cfg in ops.items():
            if not isinstance(cfg, dict):
                continue
            
            # Map op names to attribute names
            op_lower = op.lower()
            if "gate" in op_lower or "up" in op_lower:
                if "min_m" in cfg:
                    self.min_M_gate_up = int(cfg["min_m"])
                if "max_m" in cfg:
                    self.max_M_gate_up = int(cfg["max_m"])
            elif "down" in op_lower:
                if "min_m" in cfg:
                    self.min_M_down = int(cfg["min_m"])
                if "max_m" in cfg:
                    self.max_M_down = int(cfg["max_m"])
            elif any(k in op_lower for k in ["q_proj", "k_proj", "v_proj"]):
                if "min_m" in cfg:
                    self.min_M_qkv = int(cfg["min_m"])
                if "max_m" in cfg:
                    self.max_M_qkv = int(cfg["max_m"])
            elif "o" in op_lower:
                if "min_m" in cfg:
                    self.min_M_o = int(cfg["min_m"])
                if "max_m" in cfg:
                    self.max_M_o = int(cfg["max_m"])
        
        return self


    def should_use_sparse(self, op_name: str, M: int) -> bool:
        """Decide if sparse should be used for this op and batch size (efficiency frontier)"""
        if M <= 0:
            return False
        
        # Normalize op name
        op_lower = op_name.lower()
        
        # MLP ops (with max threshold)
        if "gate" in op_lower or "up" in op_lower:
            return self.min_M_gate_up <= M <= self.max_M_gate_up
        if "down" in op_lower:
            return self.min_M_down <= M <= self.max_M_down
        
        # Attention ops (with max threshold)
        if any(x in op_lower for x in ["q_proj", "k_proj", "v_proj"]):
            return self.min_M_qkv <= M <= self.max_M_qkv
        if "o_proj" in op_lower or "o" == op_lower:
            return self.min_M_o <= M <= self.max_M_o
        
        # Default fallback (no max)
        return M >= self.min_M_default