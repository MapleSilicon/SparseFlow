#!/usr/bin/env python3
import torch

def manual_24_prune(A: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    assert K % 4 == 0
    A2 = A.clone()
    blocks = A2.view(M, K // 4, 4)
    top2 = torch.topk(blocks.abs(), k=2, dim=2).indices
    mask = torch.zeros_like(blocks, dtype=torch.bool)
    mask.scatter_(2, top2, True)
    blocks[~mask] = 0
    return A2

def tflops_dense_eq(M, N, K, ms):
    flops = 2.0 * M * N * K
    return flops / (ms * 1e-3) / 1e12

def time_fn(fn, iters=200, warmup=50):
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

def ss_supported(M: int, K: int) -> bool:
    # PyTorch semi-structured constraint: both dims >= and multiple of (32, 64)
    # Here the tensor being converted is A: (M, K)
    return (M >= 32 and K >= 64 and (M % 32 == 0) and (K % 64 == 0))

@torch.inference_mode()
def main():
    if not hasattr(torch.sparse, "to_sparse_semi_structured"):
        raise SystemExit("torch.sparse.to_sparse_semi_structured not available in this torch build")

    torch.manual_seed(42)

    shapes = [
        # Attention projections (decode-ish)
        (1,   4096, 4096),
        (8,   4096, 4096),
        (32,  4096, 4096),
        (64,  4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (320, 4096, 4096),
        (384, 4096, 4096),
        (512, 4096, 4096),

        # FFN gate/up
        (128, 11008, 4096),
        (256, 11008, 4096),
        (320, 11008, 4096),
        (512, 11008, 4096),
        (1024,11008, 4096),

        # FFN down
        (128, 4096, 11008),
        (256, 4096, 11008),
        (512, 4096, 11008),
    ]

    MIN_M_SPARSE = 320
    MIN_SPEEDUP = 1.10

    print(f"{'M':>6} {'N':>6} {'K':>6} | {'Dense ms':>9} {'SS ms':>9} | {'Dense TF':>9} {'SS TF':>9} | {'Speedup':>8} {'Reco':>6} {'SS?':>4}")
    print("-" * 105)

    for (M, N, K) in shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)

        A_p = manual_24_prune(A).contiguous()

        ms_d = time_fn(lambda: A_p @ B)
        tf_d = tflops_dense_eq(M, N, K, ms_d)

        if ss_supported(M, K):
            A_ss = torch.sparse.to_sparse_semi_structured(A_p)
            ms_s = time_fn(lambda: A_ss @ B)
            tf_s = tflops_dense_eq(M, N, K, ms_s)
            sp = ms_d / ms_s
            reco = "SS" if (M >= MIN_M_SPARSE and sp >= MIN_SPEEDUP) else "DENSE"
            ssok = "YES"
        else:
            ms_s = float("nan")
            tf_s = float("nan")
            sp = float("nan")
            reco = "DENSE"
            ssok = "NO"

        ms_s_str = f"{ms_s:>9.4f}" if ssok == "YES" else f"{'NA':>9}"
        tf_s_str = f"{tf_s:>9.2f}" if ssok == "YES" else f"{'NA':>9}"
        sp_str   = f"{sp:>7.3f}x" if ssok == "YES" else f"{'NA':>8}"

        print(f"{M:>6} {N:>6} {K:>6} | {ms_d:>9.4f} {ms_s_str} | {tf_d:>9.2f} {tf_s_str} | {sp_str:>8} {reco:>6} {ssok:>4}")

        del A, B, A_p
        if ssok == "YES":
            del A_ss
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()
