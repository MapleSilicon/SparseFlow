"""
SparseFlow dispatch policy — RTX 4090, FP16, PyTorch semi-structured.
Derived from shape sweep data (2025-02).
"""

# Lower bounds: below these, sparse overhead > benefit
THRESHOLDS = {
    "attn_proj":   320,
    "ffn_gate_up": 256,
    "ffn_down":    512,
}

# Upper bounds: above these, memory/bandwidth overhead wipes out gain on Ada (SM89).
# Derived from RTX 4090 sweep (run_20260304_153802):
#   attn_proj  b=512 s=512 (M=262144) → 0.937x regression
#   ffn_gate_up b=512 s=512 (M=262144) → 1.039x marginal + OOM on accuracy
# Last clean wins at M=65536. Conservative ceiling: 131072.
# ffn_gate_up ceiling lowered to 65536: OOM on accuracy check at 131072
# on RTX 4090 (23.6GB). Timing was valid but accuracy unverified — conservative choice.
# ffn_down ceiling kept at 131072: clean PASS at all tested configs up to M=131072.
MAX_THRESHOLDS = {
    "attn_proj":   131072,
    "ffn_gate_up":  65536,
    "ffn_down":    131072,
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
    # Upper bound: beyond this M_eff, overhead beats benefit on Ada/Ampere
    max_threshold = MAX_THRESHOLDS[proj_type]
    if M_eff > max_threshold:
        return False  # upper bound gate: bandwidth-saturated regime, dense wins
    # Padded M must be valid for semi-structured
    M_padded = pad_to_multiple(M_eff, 32)
    return M_padded >= 32 and in_features % 64 == 0

def get_padded_M(M_eff: int) -> int:
    """Return the M to actually use (padded to multiple of 32)."""
    return pad_to_multiple(M_eff, 32)
