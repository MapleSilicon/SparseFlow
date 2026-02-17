import torch
import triton
import triton.language as tl

@triton.jit
def _fused_silu_mul_2d_strided(
    g_ptr, u_ptr, o_ptr,
    n_rows: tl.constexpr, n_cols: tl.constexpr,
    g_s0: tl.constexpr, g_s1: tl.constexpr,
    u_s0: tl.constexpr, u_s1: tl.constexpr,
    o_s0: tl.constexpr, o_s1: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid0 = tl.program_id(0)  # row
    pid1 = tl.program_id(1)  # col-block
    row = pid0
    col0 = pid1 * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    mask = cols < n_cols

    g_offs = row * g_s0 + cols * g_s1
    u_offs = row * u_s0 + cols * u_s1
    o_offs = row * o_s0 + cols * o_s1

    g = tl.load(g_ptr + g_offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + u_offs, mask=mask, other=0.0).to(tl.float32)

    s = 1.0 / (1.0 + tl.exp(-g))     # sigmoid(g)
    y = (g * s) * u                  # silu(g) * u

    tl.store(o_ptr + o_offs, y.to(tl.float16), mask=mask)

def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fast path: expects 2D tensors (e.g., [I, BT]) but DOES NOT require contiguity.
    Uses explicit strides so we avoid .contiguous() and the clone/copy_ tax.
    """
    assert gate.is_cuda and up.is_cuda
    assert gate.dtype == up.dtype
    assert gate.ndim == 2 and up.ndim == 2, "fused_silu_mul expects 2D tensors"

    I, BT = gate.shape
    assert up.shape == (I, BT)

    out = torch.empty((I, BT), device=gate.device, dtype=gate.dtype)

    grid = (I, triton.cdiv(BT, 1024))
    _fused_silu_mul_2d_strided[grid](
        gate, up, out,
        n_rows=I, n_cols=BT,
        g_s0=gate.stride(0), g_s1=gate.stride(1),
        u_s0=up.stride(0),   u_s1=up.stride(1),
        o_s0=out.stride(0),  o_s1=out.stride(1),
        BLOCK=1024,
        num_warps=4
    )
    return out
