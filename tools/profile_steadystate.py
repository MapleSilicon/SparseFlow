#!/usr/bin/env python3
"""Profile steady-state execution (after warmup)"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.nn.sparseflow_linear import SparseFlowLinear, make_sparseflow_linear

def replace_mlp(model, policy):
    layers = model.model.layers
    for layer in layers:
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            lin = getattr(layer.mlp, proj)
            setattr(layer.mlp, proj, make_sparseflow_linear(lin, policy=policy, op_name=proj))

def print_branch_counts(model):
    """Count sparse vs dense branch usage"""
    sparse_total = 0
    dense_total = 0
    
    for name, module in model.named_modules():
        if isinstance(module, SparseFlowLinear):
            sparse_total += module._sparse_count
            dense_total += module._dense_count
    
    print(f"\n{'='*70}")
    print("BRANCH COUNTERS")
    print(f"{'='*70}")
    print(f"Sparse branch taken: {sparse_total:,} times")
    print(f"Dense branch taken:  {dense_total:,} times")
    print(f"{'='*70}\n")

print("="*70)
print("STEADY-STATE PROFILER")
print("="*70)

# Setup
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.pad_token = tok.eos_token
batch, seq = 4, 256
inp = tok(["test"]*batch, return_tensors="pt", padding=True, 
         truncation=True, max_length=seq).to("cuda")

# Load and prepare model
print("\n1) Loading model with SparseFlow...")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    torch_dtype=torch.float16).to("cuda").eval()

policy = SparseFlowPolicy()
policy.load_runtime_policy("policy_efficient.json")
replace_mlp(model, policy)

# Compile OUTSIDE profiler
print("\n2) Compiling (this takes ~60s, NOT profiled)...")
model_c = torch.compile(model, mode="max-autotune")

# Warmup to ensure compilation done
print("\n3) Warmup (5 iterations)...")
for _ in range(5):
    _ = model_c(**inp)
torch.cuda.synchronize()

# Profile steady-state ONLY
print("\n4) Profiling steady-state (50 iterations)...")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
) as prof:
    for _ in range(50):
        _ = model_c(**inp)
    torch.cuda.synchronize()

print("\n" + "="*70)
print("CUDA KERNEL TABLE (steady-state only)")
print("="*70)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=80))

# Check for sparse kernels
table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=200)
sparse_keywords = ['sparse', 'semi_structured', 'cusparselt', 'SparseSemiStructured']
print("\n" + "="*70)
print("SPARSE KERNEL SEARCH")
print("="*70)
found_any = False
for keyword in sparse_keywords:
    if keyword.lower() in table_str.lower():
        print(f"✓ Found '{keyword}' in profile")
        found_any = True
        for line in table_str.split('\n'):
            if keyword.lower() in line.lower():
                print(f"  {line}")
if not found_any:
    print("❌ NO sparse kernels found in profile!")
    print("   This means SparseFlow sparse path is NOT being used in compiled graph")
print("="*70)

# Print branch counts
print_branch_counts(model_c)
