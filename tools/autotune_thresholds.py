#!/usr/bin/env python3
"""
Autotune SparseFlow thresholds by benchmarking crossover points.
Finds M where sparse becomes faster than pruned-dense.
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False

def prune_24(W: torch.Tensor) -> torch.Tensor:
    """2:4 pruning"""
    out_f, in_f = W.shape
    assert in_f % 4 == 0
    W4 = W.view(out_f, in_f // 4, 4)
    idx = torch.topk(W4.abs(), k=2, dim=-1, sorted=False).indices
    mask = torch.zeros_like(W4, dtype=torch.bool)
    mask.scatter_(dim=-1, index=idx, value=True)
    return torch.where(mask, W4, torch.zeros_like(W4)).view(out_f, in_f)

@torch.no_grad()
def bench(fn, iters: int, warmup: int) -> float:
    """Benchmark function in microseconds"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    
    return (time.perf_counter() - t0) * 1e6 / iters

@dataclass
class Result:
    op: str
    M: int
    pruned_dense_us: float
    sparse_weight_us: float
    speedup: float

def recommend_threshold(results: List[Result], op: str) -> int:
    """Find smallest M where speedup >= 1.05 and stays >= 1.02"""
    candidates = [r for r in results if r.op == op]
    candidates.sort(key=lambda r: r.M)
    
    for i, r in enumerate(candidates):
        if r.speedup >= 1.05:
            # Check stability (all future Ms also >= 1.02)
            tail = [x.speedup for x in candidates[i:]]
            if all(s >= 1.02 for s in tail):
                return r.M
    
    # Fallback: highest M if never wins
    return candidates[-1].M if candidates else 512

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ms", default="256,512,1024,2048,4096")
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=14336)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--out", default="benchmarks/autotune_thresholds.csv")
    args = p.parse_args()
    
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    
    Ms = [int(x.strip()) for x in args.Ms.split(",")]
    device = "cuda"
    dtype = torch.float16
    
    print("="*70)
    print("SparseFlow Threshold Autotuner")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shapes: hidden={args.hidden}, inter={args.inter}")
    print(f"Testing M values: {Ms}")
    print("="*70)
    
    # Prepare weights
    W_gate = prune_24(torch.randn(args.inter, args.hidden, device=device, dtype=dtype))
    W_gate_s = torch.sparse.to_sparse_semi_structured(W_gate)
    
    W_down = prune_24(torch.randn(args.hidden, args.inter, device=device, dtype=dtype))
    W_down_s = torch.sparse.to_sparse_semi_structured(W_down)
    
    results = []
    
    for M in Ms:
        print(f"\nTesting M={M}...")
        
        # gate/up
        X_gate = torch.randn(M, args.hidden, device=device, dtype=dtype)
        t_pd = bench(lambda: F.linear(X_gate, W_gate), args.iters, args.warmup)
        t_sp = bench(lambda: F.linear(X_gate, W_gate_s), args.iters, args.warmup)
        results.append(Result("gate_up", M, t_pd, t_sp, t_pd / t_sp))
        print(f"  gate/up: {t_pd:.1f}us → {t_sp:.1f}us ({t_pd/t_sp:.2f}×)")
        
        # down
        X_down = torch.randn(M, args.inter, device=device, dtype=dtype)
        t_pd = bench(lambda: F.linear(X_down, W_down), args.iters, args.warmup)
        t_sp = bench(lambda: F.linear(X_down, W_down_s), args.iters, args.warmup)
        results.append(Result("down", M, t_pd, t_sp, t_pd / t_sp))
        print(f"  down:    {t_pd:.1f}us → {t_sp:.1f}us ({t_pd/t_sp:.2f}×)")
    
    # Save CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "M", "pruned_dense_us", "sparse_weight_us", "speedup"])
        for r in results:
            w.writerow([r.op, r.M, f"{r.pruned_dense_us:.2f}", f"{r.sparse_weight_us:.2f}", f"{r.speedup:.3f}"])
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    for op in ["gate_up", "down"]:
        print(f"\n{op}:")
        op_results = [r for r in results if r.op == op]
        for r in sorted(op_results, key=lambda x: x.M):
            status = "✅" if r.speedup >= 1.05 else "⚠️" if r.speedup >= 1.0 else "❌"
            print(f"  M={r.M:4d}: {r.speedup:.2f}× {status}")
        
        threshold = recommend_threshold(results, op)
        print(f"  → min_M_{op} = {threshold}")
    
    print(f"\n✅ CSV saved: {args.out}")
    print("="*70)

if __name__ == "__main__":
    main()
