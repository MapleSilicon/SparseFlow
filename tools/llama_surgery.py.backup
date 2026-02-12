#!/usr/bin/env python3
"""LLaMA surgery with per-op policy"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparseflow.nn import make_sparseflow_linear
from sparseflow.nn.policy import SparseFlowPolicy


def _maybe_import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers required: {e}")


@torch.no_grad()
def replace_llama_mlp(model, policy: SparseFlowPolicy, verbose: bool = True):
    """Replace MLP linears with SparseFlowLinear (with op_name)"""
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
            
            # CRITICAL: Pass op_name so policy can differentiate
            sfl = make_sparseflow_linear(
                lin, 
                policy=policy, 
                op_name=proj_name
            )
            setattr(mlp, proj_name, sfl)
            replaced += 1
            
            if verbose:
                print(f"  [✓] Layer {li:02d} {proj_name:10s} → SparseFlowLinear (op_name={proj_name})")
    
    return replaced


@torch.no_grad()
def bench_prefill(model, tokenizer, prompt: str, batch: int, max_len: int, iters: int, warmup: int):
    device = "cuda"
    prompts = [prompt] * batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    
    for _ in range(warmup):
        _ = model(**inputs)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(**inputs)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="Explain sparse tensor cores.")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    
    AutoModelForCausalLM, AutoTokenizer = _maybe_import_transformers()
    
    print("="*70)
    print("SparseFlow Surgery (Per-Op Policy)")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Batch: {args.batch}, MaxLen: {args.max_len}, M~{args.batch*args.max_len}")
    print("="*70)
    
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
    
    print("\nApplying surgery with per-op policy...")
    policy = SparseFlowPolicy()
    print(f"Policy: gate/up_min_M={policy.min_M_gate_up}, down_min_M={policy.min_M_down}")
    
    n = replace_llama_mlp(model, policy, verbose=args.verbose)
    print(f"\n✓ Replaced {n} modules")
    
    if args.bench:
        print("\nBenchmarking...")
        ms = bench_prefill(model, tok, args.prompt, args.batch, args.max_len, args.iters, args.warmup)
        print(f"Prefill: {ms:.2f} ms")
    
    print("\n" + "="*70)
    print("Surgery complete!")
    print("="*70)


if __name__ == "__main__":
    main()
