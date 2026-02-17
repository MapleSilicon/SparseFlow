"""
SparseFlow package.

IMPORTANT:
- Keep this __init__ lightweight.
- Do NOT hard-import submodules that may depend on CUDA builds / optional ops.
- Re-export symbols *best-effort* so `import sparseflow` doesn't crash.
"""

from importlib import import_module

__all__ = []

def _safe_export(mod: str, names: list[str]) -> None:
    try:
        m = import_module(mod)
        g = globals()
        for n in names:
            if hasattr(m, n):
                g[n] = getattr(m, n)
                __all__.append(n)
    except Exception:
        # swallow import errors so package import stays healthy
        pass

# Best-effort re-exports
_safe_export("sparseflow.nn.policy", ["SparseFlowPolicy"])
_safe_export("sparseflow.nn.sparseflow_linear", ["SparseFlowLinear", "make_sparseflow_linear", "prune_24_dense_weight"])
_safe_export("sparseflow.nn.sparseflow_mlp", ["SparseFlowMLP", "make_sparseflow_mlp"])
_safe_export("sparseflow.nn.surgery", ["replace_llama_mlp_module"])
