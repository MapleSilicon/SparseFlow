#!/usr/bin/env python3
"""
GPU Gate: release sanity on a real GPU (A100/H100/RTX 30+).

Runs:
1) Quick correctness (one or two representative GEMMs)
2) Microbench thresholds for gate/up/down at key M values

Exit code != 0 if something regresses.
"""
import argparse
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = False

def _assert_cuda():
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  CC: {cc}")
    assert cc[0] >= 8, "Ampere+ required for 2:4 sparse tensor cores"

def prune_24(W: torch.Tensor) -> torch.Tensor:
    out_f, in_f = W.shape
    assert in_f % 4 == 0
    W4 = W.view(out_f, in_f // 4, 4)
    idx = torch.topk(W4.abs(), k=2, dim=-1, sorted=False).indices
    mask = torch.zeros_like(W4, dtype=torch.bool)
    mask.scatter_(dim=-1, index=idx, value=True)
    return torch.where(mask, W4, torch.zeros_like(W4)).view(out_f, in_f)

@torch.no_grad()
def bench(fn, iters=400, warmup=100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters  # us

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=14336)
    ap.add_argument("--Ms", type=str, default="256,512,1024")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--min-speedup-gateup", type=float, default=1.10)
    ap.add_argument("--min-speedup-down", type=float, default=1.10)
    args = ap.parse_args()

    _assert_cuda()
    device = "cuda"
    dtype = torch.float16

    Ms = [int(x) for x in args.Ms.split(",")]

    # Gate/Up: X[M,hidden] @ W.T[hidden,inter]
    W_gate = torch.randn(args.inter, args.hidden, device=device, dtype=dtype)
    W_gate_p = prune_24(W_gate)
    W_gate_s = torch.sparse.to_sparse_semi_structured(W_gate_p)

    # Down: X[M,inter] @ W.T[inter,hidden]
    W_down = torch.randn(args.hidden, args.inter, device=device, dtype=dtype)
    W_down_p = prune_24(W_down)
    W_down_s = torch.sparse.to_sparse_semi_structured(W_down_p)

    # Quick correctness check
    Xc = torch.randn(512, args.hidden, device=device, dtype=dtype)
    Y_dense = torch.nn.functional.linear(Xc, W_gate_p)
    Y_sparse = torch.nn.functional.linear(Xc, W_gate_s)
    err = (Y_dense.float() - Y_sparse.float()).abs().max().item()
    print(f"[correctness] gate/up max abs err vs pruned-dense: {err:.6f}")
    if err > 0.30:
        raise SystemExit(f"FAIL: correctness error too high: {err}")

    failures = 0
    print("\n=== Microbench (speedup vs pruned-dense) ===")
    for M in Ms:
        print(f"\nM={M}")
        # gate/up
        Xg = torch.randn(M, args.hidden, device=device, dtype=dtype)
        t_pd = bench(lambda: torch.nn.functional.linear(Xg, W_gate_p), args.iters, args.warmup)
        t_sp = bench(lambda: torch.nn.functional.linear(Xg, W_gate_s), args.iters, args.warmup)
        sp = t_pd / t_sp
        print(f" gate/up: {t_pd:.2f}us -> {t_sp:.2f}us  ({sp:.2f}x)")
        if M >= 256 and sp < args.min_speedup_gateup:
            print(f"  ⚠️  gate/up below threshold ({sp:.2f}x < {args.min_speedup_gateup}x)")
            failures += 1

        # down
        Xd = torch.randn(M, args.inter, device=device, dtype=dtype)
        t_pd = bench(lambda: torch.nn.functional.linear(Xd, W_down_p), args.iters, args.warmup)
        t_sp = bench(lambda: torch.nn.functional.linear(Xd, W_down_s), args.iters, args.warmup)
        sp = t_pd / t_sp
        print(f" down:    {t_pd:.2f}us -> {t_sp:.2f}us  ({sp:.2f}x)")
        if M >= 512 and sp < args.min_speedup_down:
            print(f"  ⚠️  down below threshold ({sp:.2f}x < {args.min_speedup_down}x)")
            failures += 1

    if failures:
        raise SystemExit(f"\n❌ GPU GATE FAILED: {failures} regressions detected.")
    print("\n✅ GPU GATE PASSED")

if __name__ == "__main__":
    main()
