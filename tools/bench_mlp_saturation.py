#!/usr/bin/env python3
import argparse, time, sys
sys.path.insert(0, "/workspace/SparseFlow")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparseflow.nn.policy import SparseFlowPolicy
from tools.llama_surgery_mlp import replace_llama_mlp_module

def pick_first_mlp(model):
    return model.model.layers[0].mlp

@torch.no_grad()
def time_mlp_only(mlp, hidden_states, iters=50, warmup=10):
    for _ in range(warmup):
        _ = mlp(hidden_states)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = mlp(hidden_states)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--policy", default="policy_efficient.json")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batches", default="4,16,32,64,128")
    ap.add_argument("--tokens", default="64,128,256,512")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print("="*80)
    print("MLP-ONLY SATURATION BENCHMARK")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Policy: {args.policy}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        dense = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
        sparse = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    except TypeError:
        dense = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
        sparse = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    policy = SparseFlowPolicy()
    policy.load_runtime_policy(args.policy)
    replaced = replace_llama_mlp_module(sparse, policy, verbose=False)
    print(f"Replaced: {replaced} MLP modules\n")

    dense_mlp = pick_first_mlp(dense)
    sparse_mlp = pick_first_mlp(sparse)

    H = dense.config.hidden_size
    batches = [int(x) for x in args.batches.split(",")]
    tokens = [int(x) for x in args.tokens.split(",")]

    print("Tokens | Batch | M      | Dense(ms) | Sparse(ms) | Speedup")
    print("-"*70)

    for T in tokens:
        for B in batches:
            hs = torch.randn(B, T, H, device=args.device, dtype=dtype)
            M = B * T

            td = time_mlp_only(dense_mlp, hs, iters=args.iters, warmup=args.warmup)
            ts = time_mlp_only(sparse_mlp, hs, iters=args.iters, warmup=args.warmup)

            sp = td / ts
            marker = "🎯" if sp >= 2.0 else "✓" if sp >= 1.8 else ""
            print(f"{T:6d} | {B:5d} | {M:6d} | {td:9.4f} | {ts:10.4f} | {sp:6.3f}× {marker}")

    print("-"*70)
    print("🎯 = 2.0× achieved  |  ✓ = Near 2× (1.8+)")

if __name__ == "__main__":
    main()
