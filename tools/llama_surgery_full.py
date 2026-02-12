#!/usr/bin/env python3
"""LLaMA surgery with FULL coverage (MLP + Attention)"""
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
def replace_llama_full(model, policy: SparseFlowPolicy, verbose: bool = True):
    """Replace MLP + Attention projections with SparseFlowLinear"""
    replaced = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    
    for li, layer in enumerate(layers):
        # MLP projections
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(mlp, proj_name, None)
                if lin is not None and isinstance(lin, torch.nn.Linear):
                    sfl = make_sparseflow_linear(lin, policy=policy, op_name=proj_name)
                    setattr(mlp, proj_name, sfl)
                    replaced += 1
                    if verbose:
                        print(f"  [✓] Layer {li:02d} MLP {proj_name:10s} → SparseFlowLinear")
        
        # Attention projections
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(attn, proj_name, None)
                if lin is not None and isinstance(lin, torch.nn.Linear):
                    sfl = make_sparseflow_linear(lin, policy=policy, op_name=proj_name)
                    setattr(attn, proj_name, sfl)
                    replaced += 1
                    if verbose:
                        print(f"  [✓] Layer {li:02d} ATTN {proj_name:10s} → SparseFlowLinear")
    
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
    p.add_argument("--mlp-only", action="store_true", help="Only replace MLP (old behavior)")
    args = p.parse_args()
    
    AutoModelForCausalLM, AutoTokenizer = _maybe_import_transformers()
    
    print("="*70)
    print("SparseFlow Surgery (Full Coverage: MLP + Attention)")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Batch: {args.batch}, MaxLen: {args.max_len}, M~{args.batch*args.max_len}")
    print("="*70)
    
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
    
    # Create policy with appropriate thresholds
    policy = SparseFlowPolicy(
        min_M_gate_up=256,  # MLP gate/up wins from M=256
        min_M_down=512,     # MLP down needs M=512
        min_M_qkv=512,      # Attention Q/K/V conservative
        min_M_o=512         # Attention O conservative
    )
    
    print("\nReplacing layers...")
    replaced = replace_llama_full(model, policy, verbose=args.verbose)
    print(f"\n✅ Replaced {replaced} layers with SparseFlowLinear")
    
    if args.bench:
        print(f"\nBenchmarking prefill (batch={args.batch}, seq={args.max_len})...")
        avg_time = bench_prefill(model, tok, args.prompt, args.batch, args.max_len, args.iters, args.warmup)
        print(f"  Average time: {avg_time:.2f} ms")

if __name__ == "__main__":
    main()
