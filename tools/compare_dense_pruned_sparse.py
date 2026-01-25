#!/usr/bin/env python3
"""3-way comparison with pad_token fix"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparseflow.nn.sparseflow_linear import prune_24_dense_weight, SparseFlowPolicy, SparseFlowLinear
from sparseflow.nn import make_sparseflow_linear


def _maybe_import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers required: {e}")


@torch.no_grad()
def prune_llama_mlp_inplace(model, verbose: bool = False) -> int:
    replaced = 0
    for li, layer in enumerate(getattr(model.model, "layers", [])):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, name, None)
            if isinstance(lin, nn.Linear):
                W = lin.weight.detach().to(dtype=torch.float16, device="cuda").contiguous()
                Wp = prune_24_dense_weight(W)
                lin.weight.copy_(Wp)
                replaced += 1
    return replaced


@torch.no_grad()
def sparseflow_surgery_inplace(model, policy: SparseFlowPolicy, verbose: bool = False) -> int:
    replaced = 0
    for li, layer in enumerate(getattr(model.model, "layers", [])):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, name, None)
            if isinstance(lin, nn.Linear):
                sfl = make_sparseflow_linear(lin, policy=policy, name=f"layer{li}.{name}")
                setattr(mlp, name, sfl)
                replaced += 1
    return replaced


def bench_prefill(model, tokenizer, prompt: str, batch: int, max_len: int, iters: int, warmup: int) -> float:
    device = "cuda"
    prompts = [prompt] * batch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

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
    p.add_argument("--prompt", default="Explain sparse tensor cores briefly.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--min-M", type=int, default=512)
    args = p.parse_args()

    AutoModelForCausalLM, AutoTokenizer = _maybe_import_transformers()

    print("=" * 70)
    print("3-Way Comparison: Dense vs Pruned-Dense vs Sparse")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Batch:       {args.batch}")
    print(f"MaxLen:      {args.max_len}")
    print(f"Effective M: ~{args.batch * args.max_len}")
    print(f"Policy:      min_M={args.min_M}")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    
    # Fix missing pad_token (common in Mistral/Llama)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"Set pad_token = eos_token ({tok.eos_token})")
    
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()

    # 1) Original dense
    print("\nBenchmarking original dense...")
    dense_ms = bench_prefill(base, tok, args.prompt, args.batch, args.max_len, args.iters, args.warmup)
    print(f"[1] Original Dense:       {dense_ms:.3f} ms")

    # 2) Pruned-dense
    print("\nBenchmarking pruned-dense...")
    pruned_dense = copy.deepcopy(base).to("cuda").eval()
    prune_llama_mlp_inplace(pruned_dense)
    pruned_ms = bench_prefill(pruned_dense, tok, args.prompt, args.batch, args.max_len, args.iters, args.warmup)
    print(f"[2] Pruned-Dense (2:4):   {pruned_ms:.3f} ms")

    # 3) SparseFlow
    print("\nBenchmarking SparseFlow...")
    sparse_model = copy.deepcopy(base).to("cuda").eval()
    prune_llama_mlp_inplace(sparse_model)
    policy = SparseFlowPolicy(min_M=args.min_M)
    sparseflow_surgery_inplace(sparse_model, policy=policy)
    sparse_ms = bench_prefill(sparse_model, tok, args.prompt, args.batch, args.max_len, args.iters, args.warmup)
    print(f"[3] SparseFlow (2:4):     {sparse_ms:.3f} ms")

    print("\n" + "-" * 70)
    print(f"Sparse vs Pruned-Dense:   {pruned_ms / sparse_ms:.2f}× (FAIR comparison)")
    print(f"Sparse vs Original:       {dense_ms / sparse_ms:.2f}× (includes pruning)")
    print("-" * 70)

    if args.batch * args.max_len < args.min_M:
        print(f"⚠️  M={args.batch * args.max_len} < {args.min_M}, sparse path may fallback")
    else:
        print(f"✅ M={args.batch * args.max_len} >= {args.min_M}, sparse path can activate")
    print("=" * 70)


if __name__ == "__main__":
    main()
