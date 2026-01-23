import os
import torch

# Hard-disable TF32 so we don't get hidden math mode differences.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def manual_24_prune(dense_tensor: torch.Tensor) -> torch.Tensor:
    M, K = dense_tensor.shape
    pruned = torch.zeros_like(dense_tensor)
    for i in range(M):
        for j in range(0, K, 4):
            block = dense_tensor[i, j:j+4]
            abs_vals = torch.abs(block)
            _, indices = torch.topk(abs_vals, k=2, sorted=False)
            for idx in indices:
                pruned[i, j + idx] = block[idx]
    return pruned

def stats(name: str, ref: torch.Tensor, out: torch.Tensor):
    diff = (ref - out).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    return name, max_err, mean_err

def run_case(M: int, N: int = 4096, K: int = 4096, seed: int = 42):
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')

    A_pruned = manual_24_prune(A)

    # Reference 1: fp16 output
    C_ref_fp16 = (A_pruned @ B).float()

    # Reference 2: true fp32 output reference
    C_ref_fp32 = (A_pruned.float() @ B.float())

    # Sparse path
    A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)
    C_sparse_fp16 = (A_sparse @ B).float()

    # Round-trip test
    A_roundtrip = A_sparse.to_dense()
    rt_max = (A_roundtrip - A_pruned).abs().max().item()

    # Errors
    s1 = stats("sparse_vs_ref_fp16", C_ref_fp16, C_sparse_fp16)
    s2 = stats("dense_fp16_vs_ref_fp32", C_ref_fp32, (A_pruned @ B).float())
    s3 = stats("sparse_fp16_vs_ref_fp32", C_ref_fp32, C_sparse_fp16)

    return {
        "M": M,
        "roundtrip_A_max_abs_err": rt_max,
        "sparse_vs_ref_fp16": (s1[1], s1[2]),
        "dense_fp16_vs_ref_fp32": (s2[1], s2[2]),
        "sparse_fp16_vs_ref_fp32": (s3[1], s3[2]),
    }

def main():
    print("=" * 78)
    print("SparseSemiStructured Investigation v2 (TF32 disabled)")
    print("=" * 78)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print("-" * 78)
    print("Key checks:")
    print("  (A) roundtrip_A_max_abs_err should be 0.0")
    print("  (B) If dense_fp16_vs_ref_fp32 has ~0.25, that's normal fp16 rounding")
    print("  (C) If sparse worse than dense, suspect sparse kernel bug")
    print("-" * 78)

    Ms = [64, 128, 256, 384, 512, 640, 768, 1024]
    rows = []
    for M in Ms:
        out = run_case(M)
        rows.append(out)

        a_rt = out["roundtrip_A_max_abs_err"]
        s_fp16_max, s_fp16_mean = out["sparse_vs_ref_fp16"]
        d32_max, d32_mean = out["dense_fp16_vs_ref_fp32"]
        s32_max, s32_mean = out["sparse_fp16_vs_ref_fp32"]

        print(f"M={M:4d} | A_rt={a_rt:.6f} | "
              f"sparse-vs-dense(fp16) max={s_fp16_max:.6f} mean={s_fp16_mean:.6e} | "
              f"dense(fp16)-vs-fp32 max={d32_max:.6f} | "
              f"sparse(fp16)-vs-fp32 max={s32_max:.6f}")

    print("-" * 78)
    rt_bad = [r for r in rows if r["roundtrip_A_max_abs_err"] > 0.0]
    if rt_bad:
        worst = max(rt_bad, key=lambda r: r["roundtrip_A_max_abs_err"])
        print("❌ CONVERSION ISSUE: Round-trip NOT exact")
        print(f"Worst at M={worst['M']}: err={worst['roundtrip_A_max_abs_err']:.6f}")
    else:
        print("✅ Round-trip OK: A_sparse.to_dense matches A_pruned exactly")

    worse = []
    for r in rows:
        d32_max = r["dense_fp16_vs_ref_fp32"][0]
        s32_max = r["sparse_fp16_vs_ref_fp32"][0]
        if s32_max > d32_max + 0.05:
            worse.append((r["M"], d32_max, s32_max))

    if worse:
        print("❌ SPARSE KERNEL ISSUE: Sparse worse than dense fp16 baseline")
        for M, d32, s32 in worse:
            print(f"  M={M:4d}: dense={d32:.6f} sparse={s32:.6f} (Δ={s32-d32:.6f})")
    else:
        print("✅ Sparse errors consistent with fp16 rounding/reduction order")

    print("=" * 78)

if __name__ == "__main__":
    main()
