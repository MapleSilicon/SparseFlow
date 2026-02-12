#!/usr/bin/env python3
"""Production Demo: SparseFlow with Kernel Caching"""
import sys, time, torch
sys.path.insert(0, "/workspace/sparseflow")

from transformers import AutoModelForCausalLM, AutoTokenizer
from sparseflow.nn import make_sparseflow_linear
from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.compiled_model import compile_sparseflow_model

def replace_mlp(model, policy):
    layers = model.model.layers
    for layer in layers:
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            lin = getattr(layer.mlp, proj)
            setattr(layer.mlp, proj, make_sparseflow_linear(lin, policy=policy, op_name=proj))

@torch.no_grad()
def bench(model, inputs, iters=10):
    for _ in range(5):
        _ = model(**inputs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(**inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters

print("="*70)
print("PRODUCTION DEMO: SparseFlow with Kernel Caching")
print("="*70)

# Setup
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.pad_token = tok.eos_token
inputs = tok(["test"]*4, return_tensors="pt", padding=True, 
            truncation=True, max_length=256).to("cuda")

# Load model
print("\n1) Loading model + applying SparseFlow...")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16).to("cuda").eval()

policy = SparseFlowPolicy()
policy.load_runtime_policy("policy_efficient.json")
replace_mlp(model, policy)

# Compile with caching
print("\n2) Compiling (with kernel cache)...")
compiled = compile_sparseflow_model(model, cache_dir="./sparseflow_cache")

# Benchmark
print("\n3) Benchmarking...")
t = bench(compiled, inputs)
print(f"   Time: {t:.2f}ms")

print("\n✅ DONE! Next run will load instantly from cache.")
print("="*70)
