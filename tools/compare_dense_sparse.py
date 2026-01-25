#!/usr/bin/env python3
"""Compare dense vs sparse LLaMA performance"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparseflow.nn import make_sparseflow_linear
from sparseflow.nn.sparseflow_linear import SparseFlowPolicy


def _maybe_import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers required: {e}")


@torch.no_grad()
def replace_llama_mlp(model, policy: SparseFlowPolicy):
    """Replace MLP layers with SparseFlowLinear"""
    replaced = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    
    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, proj_name, None)
            if lin is None or not isinstance(lin, torch.nn.Linear):
                continue
            
            sfl = make_sparseflow_linear(lin, policy=policy, name=f"layer{li}.{proj_name}")
            setattr(mlp, proj_name, sfl)
            replaced += 1
    
    return replaced


@torch.no_grad()
def bench_prefill(model, tokenizer, prompt: str, iters: int = 30, warmup: int = 10):
    """Benchmark prefill timing"""
    device = "cuda"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup
    for _ in range(warmup):
        _ = model(**inputs)
    torch.cuda.synchronize()
    
    # Measure
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(**inputs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    return (t1 - t0) * 1000.0 / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="Explain quantum computing in detail.")
    p.add_argument("--min-M", type=int, default=512)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    
    AutoModelForCausalLM, AutoTokenizer = _maybe_import_transformers()
    
    print("="*70)
    print("Dense vs Sparse Performance Comparison")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Policy: min_M={args.min_M}")
    print()
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_dense = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_dense.eval()
    print("✓ Model loaded")
    print()
    
    # Benchmark DENSE
    print("Benchmarking DENSE model...")
    dense_ms = bench_prefill(model_dense, tokenizer, args.prompt, args.iters, args.warmup)
    print(f"  Dense prefill: {dense_ms:.3f} ms")
    print()
    
    # Apply surgery for SPARSE
    print("Applying SparseFlow surgery...")
    policy = SparseFlowPolicy(min_M=args.min_M)
    n = replace_llama_mlp(model_dense, policy)
    print(f"  Replaced {n} modules")
    print()
    
    # Benchmark SPARSE
    print("Benchmarking SPARSE model...")
    sparse_ms = bench_prefill(model_dense, tokenizer, args.prompt, args.iters, args.warmup)
    print(f"  Sparse prefill: {sparse_ms:.3f} ms")
    print()
    
    # Results
    speedup = dense_ms / sparse_ms
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Dense:   {dense_ms:.3f} ms")
    print(f"Sparse:  {sparse_ms:.3f} ms")
    print(f"Speedup: {speedup:.2f}×")
    
    if speedup >= 1.1:
        print("✅ Significant speedup achieved!")
    elif speedup >= 1.0:
        print("✅ Slight speedup")
    else:
        print("⚠️  Slower (batch size too small or overhead dominates)")
    
    print("="*70)


if __name__ == "__main__":
    main()
