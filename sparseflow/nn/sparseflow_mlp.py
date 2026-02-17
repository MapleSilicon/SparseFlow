"""SparseFlowMLP: Optimized MLP replacement"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.nn.sparseflow_linear import prune_24_dense_weight
from sparseflow.kernels.fused_silu_mul import fused_silu_mul


class SparseFlowMLP(nn.Module):
    """Drop-in replacement for LlamaMLP using sparse tensor cores."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_bias: Optional[torch.Tensor] = None,
        up_bias: Optional[torch.Tensor] = None,
        down_bias: Optional[torch.Tensor] = None,
        policy: SparseFlowPolicy = SparseFlowPolicy(),
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        # Some HF/profiling codepaths run under torch.inference_mode().
        # Registering buffers from inference tensors can later trip:
        # "Cannot set version_counter for inference tensor".
        # Force normal tensors here.
        with torch.inference_mode(False):
            gate_weight = gate_weight.detach().clone()
            up_weight   = up_weight.detach().clone()
            down_weight = down_weight.detach().clone()

            if gate_bias is not None:
                gate_bias = gate_bias.detach().clone()
            if up_bias is not None:
                up_bias = up_bias.detach().clone()
            if down_bias is not None:
                down_bias = down_bias.detach().clone()

        # Keep original dense down_weight for the final matmul (for now)
        self.register_buffer("down_weight", down_weight, persistent=False)

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.policy = policy

        # Convert to sparse semi-structured tensors (2:4)
        gate_pruned = prune_24_dense_weight(gate_weight.contiguous())
        up_pruned   = prune_24_dense_weight(up_weight.contiguous())
        down_pruned = prune_24_dense_weight(down_weight.contiguous())

        self.register_buffer(
            "gate_sparse",
            torch.sparse.to_sparse_semi_structured(gate_pruned),
            persistent=False,
        )
        self.register_buffer(
            "up_sparse",
            torch.sparse.to_sparse_semi_structured(up_pruned),
            persistent=False,
        )
        self.register_buffer(
            "down_sparse",
            torch.sparse.to_sparse_semi_structured(down_pruned),
            persistent=False,
        )

        # Biases (may be None)
        self.register_buffer("gate_bias", gate_bias.contiguous() if gate_bias is not None else None, persistent=False)
        self.register_buffer("up_bias",   up_bias.contiguous()   if up_bias is not None else None, persistent=False)
        self.register_buffer("down_bias", down_bias.contiguous() if down_bias is not None else None, persistent=False)

    def _spmm_pick(self, Ws, xT):
        """
        Auto-pick packed/meta orientation + handle alignment.
        xT: [K, Ntokens] (transposed view)
        """
        orig_n = xT.shape[1]
        pad_val = (16 - (orig_n % 16)) % 16
        if pad_val:
            xT = F.pad(xT, (0, pad_val))

        # Ws is SparseSemiStructuredTensor
        if Ws.data.shape[1] == xT.shape[0]:
            res = torch.ops.aten._sparse_semi_structured_mm(Ws.packed, Ws.meta, xT)
        elif Ws.data.shape[0] == xT.shape[0]:
            res = torch.ops.aten._sparse_semi_structured_mm(Ws.packed_t, Ws.meta_t, xT)
        else:
            raise RuntimeError(f"Shape mismatch: Ws.data={tuple(Ws.data.shape)} xT={tuple(xT.shape)}")

        if pad_val:
            res = res[:, :orig_n]
        return res

    def forward(self, x):
        # x: [B, T, H]
        B, T, H = x.shape
        BT = B * T

        x2d = x.view(BT, H)
        xT = x2d.transpose(0, 1)  # [H, BT]

        gateT = self._spmm_pick(self.gate_sparse, xT)  # [I, BT]
        upT   = self._spmm_pick(self.up_sparse,   xT)  # [I, BT]

        if self.gate_bias is not None:
            gateT = gateT + self.gate_bias.view(-1, 1)
        if self.up_bias is not None:
            upT = upT + self.up_bias.view(-1, 1)

        hiddenT = fused_silu_mul(gateT, upT)  # [I, BT]

        # Down projection (dense for now): [H, I] @ [I, BT] -> [H, BT]
        outT = torch.matmul(self.down_weight, hiddenT)

        if self.down_bias is not None:
            outT = outT + self.down_bias.view(-1, 1)

        out2d = outT.transpose(0, 1)  # [BT, H]
        return out2d.view(B, T, H)


def make_sparseflow_mlp(*args, **kwargs):
    """Backward-compat factory for older tooling."""
    return SparseFlowMLP(*args, **kwargs)
