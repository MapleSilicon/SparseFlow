#!/usr/bin/env python3
import argparse
import torch

def manual_24_prune(A: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    assert K % 4 == 0
    A2 = A.clone()
    blocks = A2.view(M, K // 4, 4)
    absb = blocks.abs()
    top2 = torch.topk(absb, k=2, dim=2).indices
    mask = torch.zeros_like(blocks, dtype=torch.bool)
    mask.scatter_(2, top2, True)
    blocks[~mask] = 0
    return A2

def tflops_dense_eq(M, N, K, ms):
    flops = 2.0 * M * N * K
    return flops / (ms * 1e-3) / 1e12

def time_fn(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    if not hasattr(torch.sparse, "to_sparse_semi_structured"):
        raise SystemExit("torch.sparse.to_sparse_semi_structured not available in this torch build")

    dtype = torch.float16
    M,N,K = args.M, args.N, args.K
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    A_pruned = manual_24_prune(A).contiguous()
    A_ss = torch.sparse.to_sparse_semi_structured(A_pruned)

    def dense():
        return A_pruned @ B

    def ss():
        return A_ss @ B

    ms_dense = time_fn(dense, args.iters, args.warmup)
    ms_ss = time_fn(ss, args.iters, args.warmup)

    tf_dense = tflops_dense_eq(M,N,K,ms_dense)
    tf_ss_eq = tflops_dense_eq(M,N,K,ms_ss)
    print(f"M={M} N={N} K={K}")
    print(f"Dense(pruned) ms={ms_dense:.6f} TF(dense-eq)={tf_dense:.2f}")
    print(f"Semi-Structured ms={ms_ss:.6f} TF(dense-eq)={tf_ss_eq:.2f} TF(real)~{tf_ss_eq*0.5:.2f}")
    print(f"Speedup(dense-eq)={(ms_dense/ms_ss):.3f}x")

if __name__ == "__main__":
    main()
