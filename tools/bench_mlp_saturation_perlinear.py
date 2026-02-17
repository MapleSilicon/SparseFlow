#!/usr/bin/env python3
"""MLP saturation benchmark using proven per-Linear approach"""
import argparse, time, sys
sys.path.insert(0, "/workspace/SparseFlow")

import torch
from transformers import AutoModelForCausalLM

from sparseflow.nn.policy import SparseFlowPolicy
from tools.llama_surgery import replace_llama_mlp

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

print("="*80)
print("MLP-ONLY SATURATION BENCHMARK (Per-Linear Approach)")
print("="*80)

dtype = torch.float16

# Load models
try:
    dense = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=dtype).to("cuda").eval()
    sparse = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=dtype).to("cuda").eval()
except TypeError:
    dense = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=dtype).to("cuda").eval()
    sparse = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=dtype).to("cuda").eval()

# Apply sparse surgery
policy = SparseFlowPolicy()
policy.load_runtime_policy("policy_efficient.json")
replaced = replace_llama_mlp(sparse, policy, verbose=False)
print(f"Replaced: {replaced} linear layers\n")

dense_mlp = pick_first_mlp(dense)
sparse_mlp = pick_first_mlp(sparse)

H = 2048
batches = [4, 16, 32, 64, 128, 256]
tokens = [64, 128, 256, 512, 1024]

print("Tokens | Batch | M      | Dense(ms) | Sparse(ms) | Speedup")
print("-"*70)

for T in tokens:
    for B in batches:
        hs = torch.randn(B, T, H, device="cuda", dtype=dtype)
        M = B * T

        td = time_mlp_only(dense_mlp, hs, iters=50, warmup=10)
        ts = time_mlp_only(sparse_mlp, hs, iters=50, warmup=10)

        sp = td / ts
        marker = "🎯" if sp >= 2.0 else "✓" if sp >= 1.8 else ""
        print(f"{T:6d} | {B:5d} | {M:6d} | {td:9.4f} | {ts:10.4f} | {sp:6.3f}× {marker}")

print("-"*70)
print("🎯 = 2.0× achieved  |  ✓ = Near 2× (1.8+)")
print("\nNote: This is MLP-only compute. E2E will be lower due to attention.")
