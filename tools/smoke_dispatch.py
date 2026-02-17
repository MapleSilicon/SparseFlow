#!/usr/bin/env python3
"""
Smoke test: verify dispatch policy + basic correctness vs dense reference.
"""
import sys
sys.path.insert(0, "/root/sparseflow")

import torch
import torch.nn as nn

from sparseflow.dispatch import should_use_sparse, classify_projection
from sparseflow.fused_mlp import SparseFlowFusedMLP

def test_dispatch_policy():
    print("=== Dispatch Policy Tests ===")
    cases = [
        (1,    4096, 4096,  False, "decode attn"),
        (128,  4096, 4096,  False, "small attn"),
        (320,  4096, 4096,  True,  "prefill attn"),
        (512,  4096, 4096,  True,  "large attn"),

        (128,  4096, 11008, False, "small gate_up"),
        (256,  4096, 11008, True,  "medium gate_up"),
        (512,  4096, 11008, True,  "large gate_up"),

        (128, 11008, 4096,  False, "small down"),
        (256, 11008, 4096,  False, "medium down"),
        (512, 11008, 4096,  True,  "large down"),
    ]
    ok = True
    for M, inf, outf, expected, label in cases:
        got = should_use_sparse(M, inf, outf)
        status = "PASS" if got == expected else "FAIL"
        ok &= (status == "PASS")
        proj = classify_projection(inf, outf)
        dispatch = "SS" if got else "DENSE"
        print(f"  {status} M={M:>5} {inf}→{outf} ({proj:>12}) → {dispatch:>5}  [{label}]")
    return ok

def prune_24_inplace(w: torch.Tensor):
    # w: [O, K], prune along K in groups of 4 per row
    O, K = w.shape
    assert K % 4 == 0
    blocks = w.view(O, K // 4, 4)
    top2 = torch.topk(blocks.abs(), k=2, dim=2).indices
    mask = torch.zeros_like(blocks, dtype=torch.bool)
    mask.scatter_(2, top2, True)
    blocks[~mask] = 0

@torch.inference_mode()
def test_correctness():
    print("\n=== Correctness Tests (vs dense reference) ===")
    torch.manual_seed(42)
    H, I = 4096, 11008

    gate = nn.Linear(H, I, bias=False).cuda().half()
    up   = nn.Linear(H, I, bias=False).cuda().half()
    down = nn.Linear(I, H, bias=False).cuda().half()

    # Prune weights to 2:4
    prune_24_inplace(gate.weight.data)
    prune_24_inplace(up.weight.data)
    prune_24_inplace(down.weight.data)

    fused = SparseFlowFusedMLP(gate, up, down, force_mode="auto", debug=True).cuda()

    ok = True
    for M, label in [(64, "decode-ish"), (320, "prefill-edge"), (512, "prefill")]:
        x = torch.randn(M, H, device="cuda", dtype=torch.float16)

        # Dense reference (pruned weights)
        g = gate(x)
        u = up(x)
        ref = down(nn.SiLU()(g) * u)

        out = fused(x)

        err = (ref - out).abs()
        max_err = err.max().item()
        mean_err = err.mean().item()

        # FP16-equivalence tolerance: allow small quantization drift
        status = "PASS" if max_err < 0.5 else "FAIL"
        ok &= (status == "PASS")
        print(f"  {status} M={M:>5} [{label:>12}] max_err={max_err:.4f} mean_err={mean_err:.6f}")

    return ok

if __name__ == "__main__":
    p1 = test_dispatch_policy()
    p2 = test_correctness()
    print("\n" + "="*40)
    if p1 and p2:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
