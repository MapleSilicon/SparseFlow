#!/usr/bin/env python3
"""Clean compiled benchmark: Dense vs SparseFlow at seq_len=512."""
import sys
sys.path.insert(0, "/root/sparseflow")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparseflow.fused_mlp import SparseFlowFusedMLP

def prune_24(w):
    w2 = w.clone()
    blocks = w2.view(w2.shape[0], w2.shape[1] // 4, 4)
    top2 = torch.topk(blocks.abs(), k=2, dim=2).indices
    mask = torch.zeros_like(blocks, dtype=torch.bool)
    mask.scatter_(2, top2, True)
    blocks[~mask] = 0
    return w2

def time_fn(fn, iters=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

@torch.inference_mode()
def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Force exact seq_len=512
    text = "The future of artificial intelligence and computing " * 100
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    input_ids = enc["input_ids"].cuda()
    attention_mask = enc["attention_mask"].cuda()
    assert input_ids.shape[1] == 512, f"Expected 512, got {input_ids.shape[1]}"
    print(f"Seq len: {input_ids.shape[1]}")

    def fwd(m):
        return m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    # === Dense ===
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    for layer in model.model.layers:
        mlp = layer.mlp
        mlp.gate_proj.weight.data = prune_24(mlp.gate_proj.weight)
        mlp.up_proj.weight.data = prune_24(mlp.up_proj.weight)
        mlp.down_proj.weight.data = prune_24(mlp.down_proj.weight)

    ms_dense_eager = time_fn(lambda: fwd(model))
    print(f"\n1) Dense (eager):         {ms_dense_eager:.3f} ms")

    model_c = torch.compile(model, mode="reduce-overhead")
    for _ in range(5):
        fwd(model_c)
    ms_dense_compiled = time_fn(lambda: fwd(model_c))
    print(f"2) Dense (compiled):      {ms_dense_compiled:.3f} ms  ({ms_dense_eager/ms_dense_compiled:.3f}x vs eager)")

    del model, model_c
    torch.cuda.empty_cache()

    # === SparseFlow ===
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda"
    )
    model2.eval()
    for layer in model2.model.layers:
        mlp = layer.mlp
        mlp.gate_proj.weight.data = prune_24(mlp.gate_proj.weight)
        mlp.up_proj.weight.data = prune_24(mlp.up_proj.weight)
        mlp.down_proj.weight.data = prune_24(mlp.down_proj.weight)

    for layer in model2.model.layers:
        mlp = layer.mlp
        fused = SparseFlowFusedMLP(mlp.gate_proj, mlp.up_proj, mlp.down_proj).cuda().half()
        layer.mlp = fused

    ms_sf_eager = time_fn(lambda: fwd(model2))
    print(f"\n3) SparseFlow (eager):    {ms_sf_eager:.3f} ms  ({ms_dense_eager/ms_sf_eager:.3f}x vs dense eager)")

    model2_c = torch.compile(model2, mode="reduce-overhead")
    for _ in range(5):
        fwd(model2_c)
    ms_sf_compiled = time_fn(lambda: fwd(model2_c))
    print(f"4) SparseFlow (compiled): {ms_sf_compiled:.3f} ms  ({ms_dense_eager/ms_sf_compiled:.3f}x vs dense eager)")

    print(f"\n{'='*60}")
    print(f"KEY METRIC: Dense compiled vs SparseFlow compiled")
    print(f"  Dense compiled:      {ms_dense_compiled:.3f} ms")
    print(f"  SparseFlow compiled: {ms_sf_compiled:.3f} ms")
    print(f"  Speedup:             {ms_dense_compiled/ms_sf_compiled:.3f}x")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
