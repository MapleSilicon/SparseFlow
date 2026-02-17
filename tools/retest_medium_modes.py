#!/usr/bin/env python3
import sys, time, argparse, torch
sys.path.insert(0, "/workspace/SparseFlow")

from transformers import AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy

# Import both surgery styles
from tools.llama_surgery import replace_llama_mlp as replace_mlp_module_style
from tools.llama_surgery_mlp import replace_llama_mlp_perlinear as replace_mlp_perlinear_style

def pick_first_mlp(model):
    return model.model.layers[0].mlp

def pick_gate_proj(model):
    return model.model.layers[0].mlp.gate_proj

@torch.no_grad()
def time_mlp(mlp, hs, iters=80, warmup=20):
    for _ in range(warmup):
        _ = mlp(hs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = mlp(hs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters

@torch.no_grad()
def micro_profile_has_sparse_mm(mlp, hs, iters=20, warmup=10):
    from torch.profiler import profile, ProfilerActivity
    for _ in range(warmup):
        _ = mlp(hs)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            _ = mlp(hs)
        torch.cuda.synchronize()

    txt = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    return txt, ("_sparse_semi_structured_mm" in txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--policy", default="policy_efficient.json")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print("="*90)
    print(f"RETEST MEDIUM MODES (T={args.tokens}, B={args.batch}, dtype={args.dtype})")
    print("="*90)

    # Load dense baseline
    try:
        dense = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).cuda().eval()
    except TypeError:
        dense = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).cuda().eval()

    H = dense.config.hidden_size
    hs = torch.randn(args.batch, args.tokens, H, device="cuda", dtype=dtype)

    td = time_mlp(pick_first_mlp(dense), hs, iters=args.iters, warmup=args.warmup)
    print(f"\n[DENSE]  {td:.4f} ms")

    # Prepare policy
    policy = SparseFlowPolicy()
    policy.load_runtime_policy(args.policy)

    # -------------------------
    # MODE A: MODULE STYLE
    # -------------------------
    try:
        sparse_a = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).cuda().eval()
    except TypeError:
        sparse_a = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).cuda().eval()

    replaced_a = replace_mlp_module_style(sparse_a, policy, verbose=False)
    gate_a = pick_gate_proj(sparse_a)

    y_a = gate_a(hs)
    print("\n[SPARSE-MODULE] replaced:", replaced_a)
    print("[SPARSE-MODULE] gate_proj type:", type(gate_a).__name__)
    print("[SPARSE-MODULE] gate_proj out contiguous:", y_a.is_contiguous(), "stride:", y_a.stride(), "shape:", tuple(y_a.shape))

    # If SparseFlowLinear counters exist, print them
    if hasattr(gate_a, "_sparse_count"):
        print("[SPARSE-MODULE] gate_proj counts: sparse =", getattr(gate_a, "_sparse_count", None),
              "dense =", getattr(gate_a, "_dense_count", None))

    ts_a = time_mlp(pick_first_mlp(sparse_a), hs, iters=args.iters, warmup=args.warmup)
    print(f"[SPARSE-MODULE] {ts_a:.4f} ms  | speedup = {td/ts_a:.3f}×")

    txt_a, has_sparse_a = micro_profile_has_sparse_mm(pick_first_mlp(sparse_a), hs, iters=20, warmup=10)
    print("[SPARSE-MODULE] profiler has _sparse_semi_structured_mm:", has_sparse_a)

    # -------------------------
    # MODE B: PER-LINEAR STYLE
    # -------------------------
    try:
        sparse_b = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).cuda().eval()
    except TypeError:
        sparse_b = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).cuda().eval()

    replaced_b = replace_mlp_perlinear_style(sparse_b, policy, verbose=False)
    gate_b = pick_gate_proj(sparse_b)

    y_b = gate_b(hs)
    print("\n[SPARSE-PERLINEAR] replaced:", replaced_b)
    print("[SPARSE-PERLINEAR] gate_proj type:", type(gate_b).__name__)
    print("[SPARSE-PERLINEAR] gate_proj out contiguous:", y_b.is_contiguous(), "stride:", y_b.stride(), "shape:", tuple(y_b.shape))

    if hasattr(gate_b, "_sparse_count"):
        print("[SPARSE-PERLINEAR] gate_proj counts: sparse =", getattr(gate_b, "_sparse_count", None),
              "dense =", getattr(gate_b, "_dense_count", None))

    ts_b = time_mlp(pick_first_mlp(sparse_b), hs, iters=args.iters, warmup=args.warmup)
    print(f"[SPARSE-PERLINEAR] {ts_b:.4f} ms  | speedup = {td/ts_b:.3f}×")

    txt_b, has_sparse_b = micro_profile_has_sparse_mm(pick_first_mlp(sparse_b), hs, iters=20, warmup=10)
    print("[SPARSE-PERLINEAR] profiler has _sparse_semi_structured_mm:", has_sparse_b)

    print("\n--- TOP-30 CUDA (SPARSE-MODULE) ---")
    print(txt_a)
    print("\n--- TOP-30 CUDA (SPARSE-PERLINEAR) ---")
    print(txt_b)

if __name__ == "__main__":
    main()
