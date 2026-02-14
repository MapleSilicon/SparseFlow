#!/usr/bin/env python3
"""Compare SparseFlowMLP (2 transposes) vs per-Linear (6 transposes)"""
import sys, torch, time
sys.path.insert(0, "/workspace/sparseflow")

import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.nn.sparseflow_mlp import make_sparseflow_mlp
from tools.llama_surgery import replace_llama_mlp
from sparseflow.dual_compiled import force_sparse_mode

torch.set_grad_enabled(False)

def bench(model, inp, iters=50, warmup=10):
    for _ in range(warmup):
        _ = model(**inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(**inp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters

print("="*70)
print("BENCHMARK: SparseFlowMLP vs Per-Linear")
print("="*70)

inp = {"input_ids": torch.randint(0, 32000, (4, 256), device="cuda")}
policy = SparseFlowPolicy()

# 1. Dense baseline
print("\n1) Dense baseline...")
model_d = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16, device_map="cuda").eval()
model_d_c = torch.compile(model_d, mode="max-autotune")
t_dense = bench(model_d_c, inp)
print(f"   Dense: {t_dense:.3f}ms")

# 2. Per-Linear approach (6 transposes per MLP)
print("\n2) Per-Linear approach (OLD - 6 transposes)...")
model_lin = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16, device_map="cuda").eval()
replace_llama_mlp(model_lin, policy=policy, verbose=False)
force_sparse_mode(model_lin, True)
model_lin_c = torch.compile(model_lin, mode="max-autotune")
t_linear = bench(model_lin_c, inp)
print(f"   Per-Linear: {t_linear:.3f}ms")
print(f"   vs Dense: {t_dense/t_linear:.3f}×")

# 3. SparseFlowMLP approach (2 transposes per MLP)
print("\n3) SparseFlowMLP approach (NEW - 2 transposes)...")
model_mlp = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16, device_map="cuda").eval()

# Replace entire MLP modules
for layer in model_mlp.model.layers:
    layer.mlp = make_sparseflow_mlp(layer.mlp, policy=policy)

model_mlp_c = torch.compile(model_mlp, mode="max-autotune")
t_mlp = bench(model_mlp_c, inp)
print(f"   SparseFlowMLP: {t_mlp:.3f}ms")
print(f"   vs Dense: {t_dense/t_mlp:.3f}×")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Dense:          {t_dense:.3f}ms (baseline)")
print(f"Per-Linear:     {t_linear:.3f}ms ({t_dense/t_linear:.3f}× - 6 transposes)")
print(f"SparseFlowMLP:  {t_mlp:.3f}ms ({t_dense/t_mlp:.3f}× - 2 transposes)")
print(f"Improvement:    {t_linear/t_mlp:.3f}× (MLP vs Linear)")
print("="*70)

if t_dense/t_mlp > 1.05:
    print("🎉 SPARSEFLOW MLP WINS!")
elif t_dense/t_mlp > 0.95:
    print("≈ NEUTRAL (within 5%)")
else:
    print("Still slower than dense")
