"""
sparseflow.nn

Keep this module lightweight. Do NOT hard-import optional CUDA-dependent modules
or large submodules on import, otherwise `from sparseflow.nn.policy import ...`
will fail if any other file has issues.
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
        pass

_safe_export("sparseflow.nn.policy", ["SparseFlowPolicy"])
_safe_export("sparseflow.nn.surgery", ["replace_llama_mlp_module"])
_safe_export("sparseflow.nn.sparseflow_linear", ["SparseFlowLinear", "make_sparseflow_linear", "prune_24_dense_weight"])
_safe_export("sparseflow.nn.sparseflow_mlp", ["SparseFlowMLP", "make_sparseflow_mlp"])
