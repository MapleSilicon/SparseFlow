#!/usr/bin/env python3
"""Final validation: correctness + performance + coverage"""
import sys, torch, time
sys.path.insert(0, "/workspace/sparseflow")

import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy
from tools.llama_surgery import replace_llama_mlp
from sparseflow.dual_compiled import force_sparse_mode
from sparseflow.nn.sparseflow_linear import SparseFlowLinear

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
print("FINAL VALIDATION: SPARSE VS DENSE")
print("="*70)

# Setup
inp = {"input_ids": torch.randint(0, 32000, (4, 256), device="cuda")}
policy = SparseFlowPolicy()

# 1. DENSE MODE
print("\n1) Dense mode...")
model_d = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16, device_map="cuda").eval()
replace_llama_mlp(model_d, policy=policy, verbose=False)
force_sparse_mode(model_d, False)  # Force DENSE

model_d_c = torch.compile(model_d, mode="max-autotune")
t_dense = bench(model_d_c, inp)
out_dense = model_d_c(**inp).logits

print(f"   Dense: {t_dense:.3f}ms")

# Get logits for correctness check
logits_dense = out_dense.detach().clone()

# 2. SPARSE MODE
print("\n2) Sparse mode...")
model_s = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16, device_map="cuda").eval()
replace_llama_mlp(model_s, policy=policy, verbose=False)
force_sparse_mode(model_s, True)  # Force SPARSE

model_s_c = torch.compile(model_s, mode="max-autotune")
t_sparse = bench(model_s_c, inp)
out_sparse = model_s_c(**inp).logits

print(f"   Sparse: {t_sparse:.3f}ms")

# Get logits for correctness check
logits_sparse = out_sparse.detach().clone()

# 3. CORRECTNESS CHECK
print("\n" + "="*70)
print("CORRECTNESS")
print("="*70)
max_err = (logits_dense - logits_sparse).abs().max().item()
mean_err = (logits_dense - logits_sparse).abs().mean().item()
print(f"Max error:  {max_err:.6f}")
print(f"Mean error: {mean_err:.6f}")
if max_err < 1e-2:
    print("✅ PASS: Outputs match (error < 0.01)")
elif max_err < 1e-1:
    print("⚠️  CAUTION: Small differences (error < 0.1)")
else:
    print("❌ FAIL: Large differences!")

# 4. PERFORMANCE
print("\n" + "="*70)
print("PERFORMANCE")
print("="*70)
speedup = t_dense / t_sparse
print(f"Dense:   {t_dense:.3f}ms")
print(f"Sparse:  {t_sparse:.3f}ms")
print(f"Speedup: {speedup:.3f}×")

if speedup > 1.05:
    print("✅ SPARSE WINS!")
elif speedup > 0.95:
    print("≈ NEUTRAL (within 5%)")
else:
    print("❌ SPARSE SLOWER")

# 5. COVERAGE CHECK
print("\n" + "="*70)
print("COVERAGE")
print("="*70)
sf_layers = [m for m in model_s.modules() if isinstance(m, SparseFlowLinear)]
sparse_active = sum(1 for m in sf_layers if m._sparse_count > 0)
print(f"SparseFlowLinear layers: {len(sf_layers)}")
print(f"Layers with sparse>0:    {sparse_active}")
print(f"Coverage: {100*sparse_active/len(sf_layers):.1f}%")

if sparse_active == len(sf_layers):
    print("✅ FULL COVERAGE")
else:
    print(f"⚠️  {len(sf_layers)-sparse_active} layers not using sparse!")

print("="*70)
