#!/usr/bin/env python3
"""Profile the SPARSE graph specifically"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.nn.sparseflow_linear import make_sparseflow_linear, SparseFlowLinear
from sparseflow.dual_compiled import force_sparse_mode

def replace_mlp(model, policy):
    layers = model.model.layers
    for layer in layers:
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            lin = getattr(layer.mlp, proj)
            setattr(layer.mlp, proj, make_sparseflow_linear(lin, policy=policy, op_name=proj))

def count_branches(model):
    sparse_total = 0
    dense_total = 0
    for module in model.modules():
        if isinstance(module, SparseFlowLinear):
            sparse_total += module._sparse_count
            dense_total += module._dense_count
    return sparse_total, dense_total

print("="*70)
print("PROFILE SPARSE GRAPH ONLY")
print("="*70)

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.pad_token = tok.eos_token
batch, seq = 4, 256
inp = tok(["test"]*batch, return_tensors="pt", padding=True, 
         truncation=True, max_length=seq).to("cuda")

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    torch_dtype=torch.float16).to("cuda").eval()

policy = SparseFlowPolicy()
policy.load_runtime_policy("policy_efficient.json")
replace_mlp(model, policy)

# Force sparse mode
print("Forcing SPARSE mode...")
n_forced = force_sparse_mode(model, True)
print(f"  Forced {n_forced} layers to sparse")

# Compile
print("\nCompiling (mode=max-autotune)...")
model_c = torch.compile(model, mode="max-autotune")

# Warmup
print("Warmup...")
for _ in range(5):
    _ = model_c(**inp)
torch.cuda.synchronize()

# Profile
print("\nProfiling...")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
) as prof:
    for _ in range(50):
        _ = model_c(**inp)
    torch.cuda.synchronize()

print("\n" + "="*70)
print("CUDA KERNELS")
print("="*70)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

# Search for sparse kernels
table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=200)
print("\n" + "="*70)
print("SPARSE KERNEL SEARCH")
print("="*70)
sparse_keywords = ['sparse', 'semi_structured', 'cusparselt', 'SparseSemi']
found = False
for keyword in sparse_keywords:
    if keyword.lower() in table_str.lower():
        print(f"✓ Found '{keyword}'")
        found = True
if not found:
    print("❌ NO sparse kernels found!")

# Check branch counts
s, d = count_branches(model_c)
print("\n" + "="*70)
print("BRANCH COUNTERS")
print("="*70)
print(f"Sparse: {s:,}")
print(f"Dense:  {d:,}")
print("="*70)
