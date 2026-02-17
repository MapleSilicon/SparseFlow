"""
SparseFlow dispatch policy — RTX 4090, FP16, PyTorch semi-structured.
Derived from shape sweep data (2025-02).
"""

THRESHOLDS = {
    "attn_proj":   320,
    "ffn_gate_up": 256,
    "ffn_down":    512,
}

def classify_projection(in_features: int, out_features: int) -> str:
    if out_features > in_features * 2:
        return "ffn_gate_up"
    elif in_features > out_features * 2:
        return "ffn_down"
    else:
        return "attn_proj"

def pad_to_multiple(x: int, multiple: int) -> int:
    """Round up to next multiple."""
    return ((x + multiple - 1) // multiple) * multiple

def should_use_sparse(M_eff: int, in_features: int, out_features: int) -> bool:
    if M_eff < 32 or in_features % 64 != 0:
        return False
    # Check threshold against raw M_eff (before padding)
    proj_type = classify_projection(in_features, out_features)
    threshold = THRESHOLDS[proj_type]
    if M_eff < threshold:
        return False
    # Padded M must be valid for semi-structured
    M_padded = pad_to_multiple(M_eff, 32)
    return M_padded >= 32 and in_features % 64 == 0

def get_padded_M(M_eff: int) -> int:
    """Return the M to actually use (padded to multiple of 32)."""
    return pad_to_multiple(M_eff, 32)
