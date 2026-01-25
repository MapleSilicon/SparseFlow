from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_cuda_fp16(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.float16:
        raise ValueError(f"{name} must be torch.float16 (got {x.dtype})")


@torch.no_grad()
def prune_24_dense_weight(W: torch.Tensor) -> torch.Tensor:
    """Prune to 2:4 semi-structured sparsity"""
    _require_cuda_fp16(W, "W")
    if W.dim() != 2:
        raise ValueError(f"W must be 2D, got {W.dim()}D")
    out_f, in_f = W.shape
    if in_f % 4 != 0:
        raise ValueError(f"in_features must be multiple of 4 for 2:4 (got {in_f})")

    W4 = W.view(out_f, in_f // 4, 4)
    top2 = torch.topk(W4.abs(), k=2, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(W4, dtype=torch.bool)
    mask.scatter_(dim=-1, index=top2, value=True)
    return torch.where(mask, W4, torch.zeros_like(W4)).view(out_f, in_f)


@dataclass(frozen=True)
class SparseFlowPolicy:
    min_M: int = 512

    @staticmethod
    def supports_sm80() -> bool:
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8


class SparseFlowLinear(nn.Module):
    """
    Correct sparse-weight path: y = F.linear(x, W_sparse, bias)
    Avoids left-sparse transpose overhead
    """

    def __init__(
        self,
        weight_fp16: torch.Tensor,
        bias_fp16: Optional[torch.Tensor],
        policy: SparseFlowPolicy = SparseFlowPolicy(),
        name: str = "SparseFlowLinear",
    ):
        super().__init__()
        self.name = name
        self.policy = policy

        _require_cuda_fp16(weight_fp16, "weight_fp16")
        if bias_fp16 is not None:
            _require_cuda_fp16(bias_fp16, "bias_fp16")

        W_pruned = prune_24_dense_weight(weight_fp16.contiguous())
        self.register_buffer("W_dense_pruned", W_pruned, persistent=True)
        self.W_sparse = torch.sparse.to_sparse_semi_structured(self.W_dense_pruned)

        if bias_fp16 is not None:
            self.register_buffer("bias", bias_fp16.contiguous(), persistent=True)
        else:
            self.bias = None

        self.out_features, self.in_features = self.W_dense_pruned.shape

        # Probe sparse-weight support
        self._supports_sparse_weight = False
        try:
            x_probe = torch.randn(16, self.in_features, device="cuda", dtype=torch.float16)
            _ = F.linear(x_probe, self.W_sparse, self.bias)
            torch.cuda.synchronize()
            self._supports_sparse_weight = True
        except Exception:
            pass

    def extra_repr(self) -> str:
        return f"name={self.name}, in={self.in_features}, out={self.out_features}, min_M={self.policy.min_M}, sparse_weight={self._supports_sparse_weight}"

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_cuda_fp16(x, "x")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"{self.name}: expected {self.in_features}, got {x.shape[-1]}")

        leading = x.shape[:-1]
        M = 1
        for d in leading:
            M *= int(d)

        use_sparse = (
            self.policy.supports_sm80()
            and self._supports_sparse_weight
            and M >= self.policy.min_M
        )

        x2d = x.reshape(M, self.in_features).contiguous()

        # CORRECT PATH: F.linear with sparse weight
        if use_sparse:
            y2d = F.linear(x2d, self.W_sparse, self.bias)
        else:
            y2d = F.linear(x2d, self.W_dense_pruned, self.bias)

        return y2d.reshape(*leading, self.out_features)

    @property
    def supports_sparse_weight(self) -> bool:
        return self._supports_sparse_weight


@torch.no_grad()
def make_sparseflow_linear(
    dense_linear: nn.Linear,
    policy: SparseFlowPolicy = SparseFlowPolicy(),
    name: str = "SparseFlowLinear",
) -> SparseFlowLinear:
    if not isinstance(dense_linear, nn.Linear):
        raise TypeError("dense_linear must be torch.nn.Linear")
    W = dense_linear.weight.detach()
    b = dense_linear.bias.detach() if dense_linear.bias is not None else None
    return SparseFlowLinear(W, b, policy=policy, name=name)
