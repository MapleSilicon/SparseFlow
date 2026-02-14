"""
Compatibility shim.

Some tools expect:
  from sparseflow.nn.replace_llama import replace_llama_mlp, replace_llama_attention

In this repo, the implementations live under /tools:
  - tools.llama_surgery.replace_llama_mlp
  - tools.llama_surgery_full.replace_llama_full
"""

from __future__ import annotations

from tools.llama_surgery import replace_llama_mlp  # MLP-only swap
from tools.llama_surgery_full import replace_llama_full  # Full swap (MLP + attention, if implemented there)

def replace_llama_attention(model, policy, verbose: bool = True):
    """
    Attention swap entrypoint expected by older scripts.

    Current behavior: delegate to replace_llama_full(), which is the only
    "full model" surgery function exported in this repo snapshot.
    """
    return replace_llama_full(model, policy, verbose=verbose)
