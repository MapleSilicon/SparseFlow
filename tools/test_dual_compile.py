#!/usr/bin/env python3
"""Test dual compilation approach"""
import sys, torch, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.nn.sparseflow_linear import make_sparseflow_linear
from sparseflow.dual_compiled import DualCompiledModel, force_sparse_mode

def replace_mlp(model, policy):
    layers = model.model.layers
    for layer in layers:
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            lin = getattr(layer.mlp, proj)
            setattr(layer.mlp, proj, make_sparseflow_linear(lin, policy=policy, op_name=proj))

def bench(model, inp, iters=20):
    for _ in range(5):
        _ = model(**inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(**inp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters

print("="*70)
print("DUAL COMPILATION TEST")
print("="*70)

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.pad_token = tok.eos_token
policy = SparseFlowPolicy()
policy.load_runtime_policy("policy_efficient.json")

# Test at M=1024 (should use sparse)
batch, seq = 4, 256
M = batch * seq
print(f"\nTest 1: M={M} (should use SPARSE graph)")
inp = tok(["test"]*batch, return_tensors="pt", padding=True, 
         truncation=True, max_length=seq).to("cuda")

# Create two models
print("Loading models...")
model_dense = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16).to("cuda").eval()
replace_mlp(model_dense, policy)

model_sparse = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16).to("cuda").eval()
replace_mlp(model_sparse, policy)

# Force modes
print("Forcing dense mode...")
force_sparse_mode(model_dense, False)

print("Forcing sparse mode...")  
force_sparse_mode(model_sparse, True)

# Compile both
print("\nCompiling dual graphs...")
dual = DualCompiledModel(model_dense, model_sparse, policy, threshold_M=8192)
dual.compile(mode="max-autotune")

# Benchmark
print(f"\nBenchmarking M={M}...")
t = bench(dual, inp)
print(f"Time: {t:.2f}ms")

print("\n" + "="*70)
print("✅ If this works, we have dual-compiled graphs!")
print("="*70)
