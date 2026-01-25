#!/usr/bin/env python3
"""
Probe which PyTorch ops support SparseSemiStructuredTensor as WEIGHT (right operand)
"""

import time
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False

def prune_24(W: torch.Tensor) -> torch.Tensor:
    N, K = W.shape
    assert K % 4 == 0
    W4 = W.view(N, K // 4, 4)
    idx = torch.topk(W4.abs(), k=2, dim=-1, sorted=False).indices
    mask = torch.zeros_like(W4, dtype=torch.bool)
    mask.scatter_(dim=-1, index=idx, value=True)
    return torch.where(mask, W4, torch.zeros_like(W4)).view(N, K)

@torch.no_grad()
def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters

def try_op(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        return True, out
    except Exception as e:
        return False, str(e)

def main():
    device = "cuda"
    dtype = torch.float16

    M = 1024
    tests = [
        ("gate_up", M, 2048, 5632),
        ("down",    M, 5632, 2048),
    ]

    print("="*78)
    print("Probe: Sparse WEIGHT Path")
    print("="*78)
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA:  {torch.version.cuda}")
    print("-"*78)

    for tag, M, K, N in tests:
        print(f"\n[{tag}] X[{M},{K}] @ W.T[{K},{N}] → Y[{M},{N}]")

        X = torch.randn(M, K, device=device, dtype=dtype).contiguous()
        W = torch.randn(N, K, device=device, dtype=dtype).contiguous()
        Wp = prune_24(W)
        Ws = torch.sparse.to_sparse_semi_structured(Wp)

        Y_dense = F.linear(X, Wp)

        # Baseline pruned-dense
        t_dense = bench(lambda: F.linear(X, Wp))

        # Try sparse-weight in F.linear
        ok1, res1 = try_op("F.linear(X, Ws)", lambda: F.linear(X, Ws))
        if ok1:
            err = (res1 - Y_dense).abs().max().item()
            t_sp = bench(lambda: F.linear(X, Ws))
            print(f"  Pruned-Dense:           {t_dense:.2f} us")
            print(f"  Sparse-Weight(F.linear): {t_sp:.2f} us")
            print(f"  Speedup:                {t_dense/t_sp:.2f}×")
            print(f"  Max error:              {err:.6f}")
        else:
            print(f"  Pruned-Dense:           {t_dense:.2f} us")
            print(f"  Sparse-Weight(F.linear): NOT SUPPORTED")
            print(f"    Error: {res1}")

        # Try X @ Ws.t()
        ok2, res2 = try_op("X @ Ws.t()", lambda: X @ Ws.t())
        if ok2:
            err2 = (res2 - Y_dense).abs().max().item()
            t2 = bench(lambda: X @ Ws.t())
            print(f"  X @ Ws.t():             {t2:.2f} us  ({t_dense/t2:.2f}×)")
        else:
            print(f"  X @ Ws.t():             NOT SUPPORTED")

    print("\n" + "="*78)
    print("Interpretation:")
    print("  If F.linear(X, Ws) works and is faster → use that")
    print("  If NOT supported → PyTorch only does left-sparse, need custom op")
    print("="*78)

if __name__ == "__main__":
    main()
