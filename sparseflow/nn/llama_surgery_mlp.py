"""
Shim module so tooling can import:
  from sparseflow.nn.llama_surgery_mlp import replace_llama_mlp_module

Actual implementation lives in tools/llama_surgery_mlp.py
"""
import os, sys

# Ensure repo root is on path so `tools.*` is importable when sparseflow is imported as a package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.llama_surgery_mlp import replace_llama_mlp_module  # re-export

__all__ = ["replace_llama_mlp_module"]
