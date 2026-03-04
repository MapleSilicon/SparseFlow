from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

@dataclass(frozen=True)
class GateRange:
    lo: int
    hi: int

# Safe defaults for SM89 (range gate)
DEFAULT_RANGES: Dict[str, GateRange] = {
    "ffn_gate": GateRange(lo=256, hi=65_536),
    "ffn_down": GateRange(lo=512, hi=131_072),
    "attn_qkv": GateRange(lo=320, hi=131_072),
}

class ShapeOverrides:
    def __init__(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        j = json.loads(p.read_text())
        self.min_speedup = j.get("min_speedup", None)
        self.src = j.get("source_run", None)
        self.map: Dict[str, Dict[str, Any]] = j.get("overrides", {})

    def lookup(self, shape: str, batch: int, seq_len: int) -> Optional[Dict[str, Any]]:
        return self.map.get(f"{shape}|b={batch}|s={seq_len}")

def decide(shape: str, batch: int, seq_len: int,
           overrides: Optional[ShapeOverrides] = None,
           ranges: Dict[str, GateRange] = DEFAULT_RANGES) -> Tuple[bool, str]:
    # 1) Overrides decide first (max-speed mode)
    if overrides is not None:
        rec = overrides.lookup(shape, batch, seq_len)
        if rec is not None:
            if rec.get("decision") == "sparse":
                return True, f"override:sparse ({rec.get('reason','')})"
            return False, f"override:dense ({rec.get('reason','')})"

    # 2) Range gate fallback (production-safe)
    r = ranges.get(shape)
    if r is None:
        return False, "no-range:default-dense"

    m_eff = batch * seq_len
    if m_eff < r.lo:
        return False, f"gate:skip (M_eff={m_eff} < lo={r.lo})"
    if m_eff > r.hi:
        return False, f"gate:skip (M_eff={m_eff} > hi={r.hi})"
    return True, f"gate:run (lo={r.lo} <= M_eff={m_eff} <= hi={r.hi})"
