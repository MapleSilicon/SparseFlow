#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace/SparseFlow")

import argparse
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparseflow.nn.policy import SparseFlowPolicy
from tools.llama_surgery_mlp import replace_llama_mlp_perlinear, replace_llama_mlp_module

def pick_first_mlp(model):
    return model.model.layers[0].mlp

@torch.no_grad()
def run_profile(mlp, hs, iters=80, warmup=20):
    for _ in range(warmup):
        _ = mlp(hs)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            _ = mlp(hs)
        torch.cuda.synchronize()
    return prof

def load_model(model_id: str, dtype: torch.dtype):
    # Transformers compat: dtype=... vs torch_dtype=...
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).cuda().eval()
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).cuda().eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--policy", default="policy_efficient.json")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--mode", default="perlinear", choices=["perlinear", "module"],
                    help="perlinear = 66 linear layers, module = 22 MLP modules")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"Loading models for profiling (T={args.tokens}, B={args.batch})...")
    dense = load_model(args.model, dtype)
    sparse = load_model(args.model, dtype)

    policy = SparseFlowPolicy()
    policy.load_runtime_policy(args.policy)

    if args.mode == "perlinear":
        replaced = replace_llama_mlp_perlinear(sparse, policy, verbose=False)
        mode_str = "Per-Linear"
    else:
        replaced = replace_llama_mlp_module(sparse, policy, verbose=False)
        mode_str = "Module"

    print(f"Mode: {mode_str} | Replaced: {replaced}\n")

    H = dense.config.hidden_size
    hs = torch.randn(args.batch, args.tokens, H, device="cuda", dtype=dtype)

    M = args.tokens * args.batch

    print("=" * 80)
    print(f"DENSE MLP PROFILE (mode={mode_str}, T={args.tokens}, B={args.batch}, M={M})")
    print("=" * 80)
    prof_d = run_profile(pick_first_mlp(dense), hs, iters=args.iters, warmup=args.warmup)
    print(prof_d.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    print("\n" + "=" * 80)
    print(f"SPARSE MLP PROFILE (mode={mode_str}, T={args.tokens}, B={args.batch}, M={M})")
    print("=" * 80)
    prof_s = run_profile(pick_first_mlp(sparse), hs, iters=args.iters, warmup=args.warmup)
    print(prof_s.key_averages().table(sort_by="cuda_time_total", row_limit=25))

if __name__ == "__main__":
    main()
