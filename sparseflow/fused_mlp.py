"""
SparseFlow Fused MLP — zero-allocation padding via pre-allocated buffers.
"""
import torch
import torch.nn as nn
from sparseflow.dispatch import should_use_sparse, get_padded_M

class SparseFlowFusedMLP(nn.Module):
    def __init__(self, gate_proj, up_proj, down_proj, act_fn=nn.SiLU(), force_mode="auto"):
        super().__init__()
        self.act_fn = act_fn
        self.force_mode = force_mode
        self.hidden_size = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features

        fused_weight = torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0).contiguous()
        self.fused_gate_up_weight = nn.Parameter(fused_weight, requires_grad=False)
        self.down_weight = nn.Parameter(down_proj.weight.data.contiguous(), requires_grad=False)

        self._gate_up_sparse = None
        try:
            self._gate_up_sparse = torch.sparse.to_sparse_semi_structured(self.fused_gate_up_weight)
        except (RuntimeError, ValueError):
            pass

        self._down_sparse = None
        try:
            self._down_sparse = torch.sparse.to_sparse_semi_structured(self.down_weight)
        except (RuntimeError, ValueError):
            pass

        self.down_bias = down_proj.bias

        # Pre-allocated pad buffers (lazily sized)
        self._pad_buf_gu = None
        self._pad_buf_down = None

    def _get_pad_buffer(self, M_pad, K, which):
        """Return a pre-allocated zero buffer, resizing only when needed."""
        if which == "gu":
            if self._pad_buf_gu is None or self._pad_buf_gu.shape[0] < M_pad or self._pad_buf_gu.shape[1] != K:
                self._pad_buf_gu = torch.zeros(M_pad, K, device=self.fused_gate_up_weight.device,
                                                dtype=self.fused_gate_up_weight.dtype)
            return self._pad_buf_gu
        else:
            if self._pad_buf_down is None or self._pad_buf_down.shape[0] < M_pad or self._pad_buf_down.shape[1] != K:
                self._pad_buf_down = torch.zeros(M_pad, K, device=self.down_weight.device,
                                                  dtype=self.down_weight.dtype)
            return self._pad_buf_down

    def _sparse_mm_padded(self, x_2d, sparse_w, M_eff, which):
        M_pad = get_padded_M(M_eff)
        if M_pad == M_eff:
            return torch.mm(x_2d, sparse_w.t())

        K = x_2d.shape[1]
        buf = self._get_pad_buffer(M_pad, K, which)
        buf[:M_eff].copy_(x_2d)
        buf[M_eff:M_pad].zero_()
        out = torch.mm(buf[:M_pad], sparse_w.t())
        return out[:M_eff]

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.view(-1, self.hidden_size)
        M_eff = x_2d.shape[0]
        in_gu = self.fused_gate_up_weight.shape[1]
        out_gu = self.fused_gate_up_weight.shape[0]
        in_d = self.down_weight.shape[1]
        out_d = self.down_weight.shape[0]

        use_sparse_gu = (
            self.force_mode == "sparse" or
            (self.force_mode == "auto" and self._gate_up_sparse is not None
             and should_use_sparse(M_eff, in_gu, out_gu))
        )
        if use_sparse_gu:
            fused_out = self._sparse_mm_padded(x_2d, self._gate_up_sparse, M_eff, "gu")
        else:
            fused_out = nn.functional.linear(x_2d, self.fused_gate_up_weight)

        gate, up = fused_out.chunk(2, dim=-1)
        hidden = self.act_fn(gate) * up

        use_sparse_d = (
            self.force_mode == "sparse" or
            (self.force_mode == "auto" and self._down_sparse is not None
             and should_use_sparse(M_eff, in_d, out_d))
        )
        if use_sparse_d:
            out = self._sparse_mm_padded(hidden, self._down_sparse, M_eff, "down")
        else:
            out = nn.functional.linear(hidden, self.down_weight)

        if self.down_bias is not None:
            out = out + self.down_bias

        return out.view(*orig_shape[:-1], out.shape[-1])
