"""SparseFlow: Hardware-aware sparse inference for A100"""

from sparseflow.nn.sparseflow_linear import SparseFlowLinear, make_sparseflow_linear, prune_24_dense_weight
from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.compiled_model import compile_sparseflow_model, CompiledSparseFlowModel

__version__ = "2.2.0.post1"

__all__ = [
    'SparseFlowLinear',
    'make_sparseflow_linear', 
    'SparseFlowPolicy',
    'prune_24_dense_weight',
    'compile_sparseflow_model',
    'CompiledSparseFlowModel',
]
