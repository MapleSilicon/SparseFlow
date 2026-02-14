#!/usr/bin/env python3
"""Check if Per-Linear outlier is real or fluke"""
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

inp = {"input_ids": torch.randint(0, 32000, (4, 256), device="cuda")}
policy = SparseFlowPolicy()

print("="*70)
print("PER-LINEAR STABILITY CHECK (N=20)")
print("="*70)

times = []
for i in range(20):
    # Fresh model each time to avoid caching artifacts
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16, device_map="cuda").eval()
    replace_llama_mlp(model, policy=policy, verbose=False)
    force_sparse_mode(model, True)
    model_c = torch.compile(model, mode="max-autotune")
    
    t = bench(model_c, inp)
    times.append(t)
    print(f"Trial {i+1:2d}: {t:.3f} ms")

print("="*70)
print(f"Min:  {min(times):.3f} ms")
print(f"Mean: {sum(times)/len(times):.3f} ms")
print(f"Max:  {max(times):.3f} ms")
print(f"Range: {max(times)-min(times):.3f} ms")
print("="*70)

if max(times) - min(times) > 2.0:
    print("⚠️  UNSTABLE: Range > 2ms, has scheduling/sync issues")
else:
    print("✅ STABLE: Range < 2ms, outlier was one-off")
