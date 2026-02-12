#!/usr/bin/env python3
"""Compare MLP-only vs Full coverage"""
import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparseflow.nn import make_sparseflow_linear
from sparseflow.nn.policy import SparseFlowPolicy

def replace_mlp_only(model, policy):
    replaced = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(mlp, proj_name, None)
                if lin and isinstance(lin, torch.nn.Linear):
                    setattr(mlp, proj_name, make_sparseflow_linear(lin, policy=policy, op_name=proj_name))
                    replaced += 1
    return replaced

def replace_full(model, policy):
    replaced = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer in layers:
        # MLP
        mlp = getattr(layer, "mlp", None)
        if mlp:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(mlp, proj_name, None)
                if lin and isinstance(lin, torch.nn.Linear):
                    setattr(mlp, proj_name, make_sparseflow_linear(lin, policy=policy, op_name=proj_name))
                    replaced += 1
        # Attention
        attn = getattr(layer, "self_attn", None)
        if attn:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(attn, proj_name, None)
                if lin and isinstance(lin, torch.nn.Linear):
                    setattr(attn, proj_name, make_sparseflow_linear(lin, policy=policy, op_name=proj_name))
                    replaced += 1
    return replaced

@torch.no_grad()
def bench(model, tok, batch, max_len, iters=10, warmup=3):
    inputs = tok(["test"]*batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
    for _ in range(warmup):
        _ = model(**inputs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(**inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
batch, max_len = 4, 256

print("="*70)
print("Coverage Comparison: MLP-only vs Full (MLP+Attention)")
print("="*70)

tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

policy = SparseFlowPolicy(min_M_gate_up=256, min_M_down=512, min_M_qkv=512, min_M_o=512)

# Dense baseline
print("\n1) Dense baseline...")
model_dense = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda").eval()
t_dense = bench(model_dense, tok, batch, max_len)
print(f"   Time: {t_dense:.2f} ms")
del model_dense
torch.cuda.empty_cache()

# MLP-only
print("\n2) SparseFlow MLP-only (66 layers)...")
model_mlp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda").eval()
n_mlp = replace_mlp_only(model_mlp, policy)
print(f"   Replaced: {n_mlp} layers")
t_mlp = bench(model_mlp, tok, batch, max_len)
print(f"   Time: {t_mlp:.2f} ms")
print(f"   Speedup: {t_dense/t_mlp:.2f}×")
del model_mlp
torch.cuda.empty_cache()

# Full coverage
print("\n3) SparseFlow Full (154 layers)...")
model_full = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda").eval()
n_full = replace_full(model_full, policy)
print(f"   Replaced: {n_full} layers")
t_full = bench(model_full, tok, batch, max_len)
print(f"   Time: {t_full:.2f} ms")
print(f"   Speedup: {t_dense/t_full:.2f}×")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Dense:        {t_dense:.2f} ms (baseline)")
print(f"MLP-only:     {t_mlp:.2f} ms ({t_dense/t_mlp:.2f}×)")
print(f"Full:         {t_full:.2f} ms ({t_dense/t_full:.2f}×)")
print(f"Improvement:  {t_mlp/t_full:.2f}× faster with full coverage")
print("="*70)
