"""
SparseFlow SM89 (RTX 4090) Shape Gate

- Enforces BOTH lower and upper M_eff bounds (M_eff = batch * seq_len).
- Produces correct, non-embarrassing skip reasons.
- Optional per-shape overrides for "max speed mode".
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

@dataclass(frozen=True)
class GateRange:
    lo: int
    hi: int

# Defaults from SparseFlow_Benchmark_Report_Final.docx (Run 20260304_160101)
# ffn_gate_up: 256..65536
# ffn_down:    512..131072
# attn_proj:   320..131072
DEFAULT_RANGES: Dict[str, GateRange] = {
    "ffn_gate": GateRange(lo=256, hi=65_536),
    "ffn_down": GateRange(lo=512, hi=131_072),
    "attn_qkv": GateRange(lo=320, hi=131_072),
}

class ShapeOverrides:
    """
    Optional per-shape decision map:
      key:  "<op>|b=<batch>|s=<seq>"
      val:  {"decision": "dense"|"sparse", "reason": "..."}
    """
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.map: Dict[str, Dict[str, Any]] = {}
        if path:
            p = Path(path)
            if p.exists():
                j = json.loads(p.read_text())
                self.map = j.get("overrides", {}) if isinstance(j, dict) else {}

    def lookup(self, op: str, batch: int, seq: int) -> Optional[Dict[str, Any]]:
        return self.map.get(f"{op}|b={batch}|s={seq}")

def decide_sparse_sm89(
    op: str,
    batch: int,
    seq: int,
    overrides: Optional[ShapeOverrides] = None,
    ranges: Dict[str, GateRange] = DEFAULT_RANGES,
) -> Tuple[bool, str]:
    """
    Returns: (use_sparse, reason)

    Reason is always consistent with decision.
    """
    # 1) Overrides (max speed mode)
    if overrides:
        rec = overrides.lookup(op, batch, seq)
        if rec:
            dec = rec.get("decision", "dense")
            why = rec.get("reason", "override")
            if dec == "sparse":
                return True, f"override:sparse ({why})"
            return False, f"override:dense ({why})"

    # 2) Range gate (production-safe)
    r = ranges.get(op)
    if r is None:
        return False, "no-range:default-dense"

    m_eff = int(batch) * int(seq)
    if m_eff < r.lo:
        return False, f"gate:skip (M_eff={m_eff} < lo={r.lo})"
    if m_eff > r.hi:
        return False, f"gate:skip (M_eff={m_eff} > hi={r.hi})"
    return True, f"gate:run (lo={r.lo} <= M_eff={m_eff} <= hi={r.hi})"
