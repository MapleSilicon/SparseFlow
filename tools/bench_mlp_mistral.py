#!/usr/bin/env python3
"""
Microbench Mistral MLP GEMMs (apples-to-apples):

For hidden=4096, inter=14336:
  gate/up:  X[M,4096]  @ W[14336,4096]^T -> [M,14336]
  down:     X[M,14336] @ W[4096,14336]^T -> [M,4096]

We compare:
  1) Pruned-Dense (2:4 pruned weights, dense matmul)
  2) Sparse-Weight (2:4 pruned weights, SparseSemiStructuredTensor via F.linear)

This tells you crossover M where sparse starts winning.
"""

import argparse
import time
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False

def prune_24(W: torch.Tensor) -> torch.Tensor:
    out_f, in_f = W.shape
    assert in_f % 4 == 0
    W4 = W.view(out_f, in_f // 4, 4)
    idx = torch.topk(W4.abs(), k=2, dim=-1, sorted=False).indices
    mask = torch.zeros_like(W4, dtype=torch.bool)
    mask.scatter_(dim=-1, index=idx, value=True)
    return torch.where(mask, W4, torch.zeros_like(W4)).view(out_f, in_f)

@torch.no_grad()
def bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters  # us

@torch.no_grad()
def run_case(name: str, X: torch.Tensor, W_dense: torch.Tensor, iters: int, warmup: int):
    # prune once
    Wp = prune_24(W_dense.contiguous())
    Ws = torch.sparse.to_sparse_semi_structured(Wp)

    # dense on pruned weights
    t_pruned = bench(lambda: F.linear(X, Wp, None), iters, warmup)

    # sparse weight path
    t_sparse = bench(lambda: F.linear(X, Ws, None), iters, warmup)

    # correctness vs FP32 ref (sanity)
    Y_ref = (X.float() @ Wp.t().float())
    Y_s = F.linear(X, Ws, None).float()
    max_err = (Y_ref - Y_s).abs().max().item()

    sp = t_pruned / t_sparse
    print(f"[{name}]")
    print(f"  Pruned-Dense:    {t_pruned:8.2f} us")
    print(f"  Sparse-Weight:   {t_sparse:8.2f} us")
    print(f"  Speedup:         {sp:8.2f}×  (vs pruned-dense)")
    print(f"  Max error:       {max_err:8.6f}")
    return sp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=14336)
    ap.add_argument("--Ms", type=str, default="256,512,1024,2048,4096",
                    help="Comma list of M values to test")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float16

    print("="*78)
    print("SparseFlow Microbench: Mistral MLP GEMMs (pruned-dense vs sparse-weight)")
    print("="*78)
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA:  {torch.version.cuda}")
    print(f"hidden={args.hidden} inter={args.inter}")
    print("-"*78)

    Ms = [int(x.strip()) for x in args.Ms.split(",") if x.strip()]
    results = []

    # Create weights once (simulating fixed layer weights)
    W_gate = torch.randn(args.inter, args.hidden, device=device, dtype=dtype)
    W_down = torch.randn(args.hidden, args.inter, device=device, dtype=dtype)

    for M in Ms:
        print(f"\nM={M}")
        X_gate = torch.randn(M, args.hidden, device=device, dtype=dtype)
        X_down = torch.randn(M, args.inter, device=device, dtype=dtype)

        sp_gate = run_case("gate/up  X[M,hidden] -> [M,inter]", X_gate, W_gate, args.iters, args.warmup)
        sp_down = run_case("down     X[M,inter]  -> [M,hidden]", X_down, W_down, args.iters, args.warmup)

        results.append((M, sp_gate, sp_down))

    print("\n" + "="*78)
    print("Summary (speedup vs pruned-dense):")
    for M, sg, sd in results:
        print(f"  M={M:5d}  gate/up={sg:5.2f}×   down={sd:5.2f}×")
    print("="*78)

if __name__ == "__main__":
    main()
