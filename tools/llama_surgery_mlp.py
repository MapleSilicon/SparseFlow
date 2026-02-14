#!/usr/bin/env python3
"""LLaMA surgery - replace entire MLP modules (not individual linears)"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparseflow.nn.sparseflow_mlp import make_sparseflow_mlp
from sparseflow.nn.policy import SparseFlowPolicy


def _maybe_import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers required: {e}")


@torch.no_grad()
def replace_llama_mlp_module(model, policy: SparseFlowPolicy, verbose: bool = True):
    """
    Replace entire MLP modules with SparseFlowMLP.
    
    This is cleaner than replacing 66 individual linears:
    - 22 module swaps instead of 66
    - 2 transposes per block instead of 6
    - Fewer graph breaks
    - Faster compilation
    """
    replaced = 0
    layers = getattr(getattr(model, "model", model), "layers", [])

    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        # Replace entire MLP module
        sparse_mlp = make_sparseflow_mlp(mlp, policy=policy)
        layer.mlp = sparse_mlp
        replaced += 1

        if verbose:
            print(f"  [✓] Layer {li:02d} mlp → SparseFlowMLP (2 transposes)")

    if verbose:
        print(f"\nReplaced {replaced} MLP modules (vs {replaced*3} individual linears)")
        print(f"Transpose reduction: {replaced*6} → {replaced*2} per forward")

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


def replace_llama_mlp_perlinear(model, policy: SparseFlowPolicy, verbose: bool = True):
    """
    Replace individual MLP linears (gate/up/down) with SparseFlowLinear.
    This gives 3 replacements per layer → 66 for TinyLlama (22 layers).
    """
    from sparseflow.nn.sparseflow_linear import make_sparseflow_linear

    n = 0
    layers = model.model.layers
    for li, layer in enumerate(layers):
        mlp = layer.mlp
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, proj)
            setattr(mlp, proj, make_sparseflow_linear(lin, policy=policy, op_name=f"layer{li}.{proj}"))
            n += 1
        if verbose:
            print(f"  [✓] Layer {li:02d} mlp linears → SparseFlowLinear (3 linears)")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--prompt", default="Explain sparse tensor cores.")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--mode", type=str, default="mlp", choices=["dense","perlinear","mlp"],
                   help="dense=no surgery, perlinear=66 SparseFlowLinear, mlp=22 SparseFlowMLP")

    args = p.parse_args()

    AutoModelForCausalLM, AutoTokenizer = _maybe_import_transformers()

    print("="*70)
    print("SparseFlow Surgery - MLP Module Replacement")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Batch: {args.batch}, MaxLen: {args.max_len}, M={args.batch*args.max_len}")
    print("="*70)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()

    print("\nApplying MLP module surgery...")
    policy = SparseFlowPolicy()
    if args.mode == "dense":
        print("Dense mode: no surgery applied")
        n = 0
    elif args.mode == "perlinear":
        print("Applying Per-Linear surgery (SparseFlowLinear on gate/up/down)...")
        n = replace_llama_mlp_perlinear(model, policy, verbose=args.verbose)
    else:
        print("Applying MLP module surgery (SparseFlowMLP)...")
        n = replace_llama_mlp_module(model, policy, verbose=args.verbose)
    if args.bench:
        print("\nBenchmarking...")
        ms = bench_prefill(model, tok, args.prompt, args.batch, args.max_len, args.iters, args.warmup)
        print(f"Prefill: {ms:.2f} ms")

    print("\n" + "="*70)
    print("Surgery complete!")
    print("="*70)


if __name__ == "__main__":
    main()
