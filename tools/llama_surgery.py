#!/usr/bin/env python3
"""
LLaMA Model Surgery Tool
Replaces FFN projections (gate_proj, up_proj, down_proj) with SparseFlowLinear
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Add parent dir to path for sparseflow import
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparseflow.nn import make_sparseflow_linear
from sparseflow.nn.sparseflow_linear import SparseFlowPolicy


def _maybe_import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            f"transformers is required. Install: pip install transformers\nError: {e}"
        )


@torch.no_grad()
def replace_llama_mlp_linears(model, policy: SparseFlowPolicy, verbose: bool = True) -> int:
    """
    Replace LLaMA MLP linears (gate_proj, up_proj, down_proj) with SparseFlowLinear.
    
    Returns:
        Number of replaced modules
    """
    replaced = 0

    # HuggingFace LLaMA: model.model.layers[i].mlp.{gate_proj,up_proj,down_proj}
    layers = getattr(getattr(model, "model", model), "layers", [])
    
    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, proj_name, None)
            if lin is None or not isinstance(lin, torch.nn.Linear):
                continue

            # Convert to SparseFlowLinear
            sfl = make_sparseflow_linear(
                lin, 
                policy=policy, 
                name=f"layer{li}.mlp.{proj_name}"
            )
            setattr(mlp, proj_name, sfl)
            replaced += 1

            if verbose:
                W_shape = lin.weight.shape
                print(f"  [✓] Layer {li:02d} {proj_name:10s} {W_shape} → SparseFlowLinear")

    return replaced


@torch.no_grad()
def smoke_forward(model, tokenizer, prompt: str, max_new_tokens: int = 1) -> None:
    """Quick smoke test - single forward pass"""
    device = "cuda"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Single forward (prefill)
    _ = model(**inputs)
    
    # Tiny generate for sanity
    _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)


@torch.no_grad()
def bench_prefill(model, tokenizer, prompt: str, iters: int = 20, warmup: int = 5) -> float:
    """Benchmark prefill (forward pass only)"""
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
    p = argparse.ArgumentParser(description="Apply SparseFlow to LLaMA FFN layers")
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--prompt", default="Explain 2:4 structured sparsity in one sentence.", 
                   help="Test prompt")
    p.add_argument("--min-M", type=int, default=512, 
                   help="Sparse enable threshold (batch*seq >= min_M)")
    p.add_argument("--bench", action="store_true", 
                   help="Run prefill benchmark")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    AutoModelForCausalLM, AutoTokenizer = _maybe_import_transformers()

    print("=" * 70)
    print("SparseFlow LLaMA Surgery")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Policy: min_M={args.min_M}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"✓ Model loaded ({model.config.model_type})")
    print()

    # Apply surgery
    policy = SparseFlowPolicy(min_M=args.min_M)
    print("Replacing MLP layers with SparseFlowLinear...")
    n = replace_llama_mlp_linears(model, policy=policy, verbose=args.verbose)
    print(f"\n✓ Replaced {n} modules with SparseFlowLinear")
    print()

    # Smoke test
    print("Running smoke test...")
    smoke_forward(model, tokenizer, args.prompt, max_new_tokens=1)
    print("✓ Smoke test passed")
    print()

    # Optional benchmark
    if args.bench:
        print(f"Benchmarking prefill ({args.iters} iters, {args.warmup} warmup)...")
        ms = bench_prefill(model, tokenizer, args.prompt, iters=args.iters, warmup=args.warmup)
        print(f"✓ Prefill avg: {ms:.3f} ms")
        print()

    print("=" * 70)
    print("Surgery complete! Model ready for inference.")
    print("=" * 70)


if __name__ == "__main__":
    main()
