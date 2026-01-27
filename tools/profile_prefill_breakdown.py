#!/usr/bin/env python3
"""
Prefill breakdown profiler - FIXED with per-call CUDA events
"""

import argparse
from collections import defaultdict
from typing import Dict, Tuple

import torch
import torch.nn as nn


def _maybe_import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers required: {e}")


# Global storage for current iteration's events
_iter_events = []


def pre_hook(mod, inp):
    """Create fresh events for this call"""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    mod.__sf_evt = (s, e)


def post_hook(mod, inp, out):
    """Record end event and store"""
    s, e = mod.__sf_evt
    e.record()
    _iter_events.append((mod.__sf_cat, s, e))


def classify_linear(name: str) -> str:
    """Categorize Linear layers"""
    n = name.lower()
    if "gate_proj" in n:
        return "MLP_gate"
    if "up_proj" in n:
        return "MLP_up"
    if "down_proj" in n:
        return "MLP_down"
    if "q_proj" in n:
        return "Attn_q"
    if "k_proj" in n:
        return "Attn_k"
    if "v_proj" in n:
        return "Attn_v"
    if "o_proj" in n:
        return "Attn_o"
    return "Other"


def attach_hooks(model) -> list:
    """Attach per-call event hooks"""
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            mod.__sf_cat = classify_linear(name)
            handles.append(mod.register_forward_pre_hook(pre_hook))
            handles.append(mod.register_forward_hook(post_hook))
    return handles


@torch.no_grad()
def run_profile(model, inputs, iters: int, warmup: int) -> Tuple[float, Dict, Dict]:
    """Run profiling with per-call event collection"""
    global _iter_events
    
    # Warmup
    for _ in range(warmup):
        model(**inputs, use_cache=False)
    torch.cuda.synchronize()
    
    stats_ms = defaultdict(float)
    stats_calls = defaultdict(int)
    total_ms = 0.0
    
    for i in range(iters):
        _iter_events = []
        
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        
        model(**inputs, use_cache=False)
        
        t1.record()
        torch.cuda.synchronize()
        
        total_ms += t0.elapsed_time(t1)
        
        # Debug first iteration
        if i == 0:
            print(f"\n[DEBUG] Events captured in first iteration: {len(_iter_events)}")
        
        # Accumulate per-call timings
        for cat, s, e in _iter_events:
            stats_ms[cat] += s.elapsed_time(e)
            stats_calls[cat] += 1
    
    # Average per iteration
    total_ms /= iters
    for k in list(stats_ms.keys()):
        stats_ms[k] /= iters
        stats_calls[k] = int(stats_calls[k] / iters)
    
    return total_ms, stats_ms, stats_calls


def print_breakdown(total_ms: float, stats_ms: Dict, stats_calls: Dict):
    """Print timing breakdown"""
    print("\n" + "="*70)
    print("PREFILL BREAKDOWN (Per-Call CUDA Events)")
    print("="*70)
    print(f"Total prefill: {total_ms:.2f} ms\n")
    
    # Sort by time
    rows = [(k, v, stats_calls[k]) for k, v in stats_ms.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    
    total_hooked = sum(r[1] for r in rows)
    
    print(f"{'Category':<15} {'Time (ms)':>12} {'Calls/iter':>10} {'% Total':>10}")
    print("-"*70)
    
    for k, ms, calls in rows:
        pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
        print(f"{k:<15} {ms:>12.2f} {calls:>10d} {pct:>9.1f}%")
    
    print("-"*70)
    unaccounted = total_ms - total_hooked
    unaccounted_pct = (unaccounted / total_ms * 100.0) if total_ms > 0 else 0.0
    print(f"{'Linear total':<15} {total_hooked:>12.2f} {'':>10} {total_hooked/total_ms*100.0:>9.1f}%")
    print(f"{'Other ops':<15} {unaccounted:>12.2f} {'':>10} {unaccounted_pct:>9.1f}%")
    print("="*70)
    
    # Aggregate
    mlp = sum(r[1] for r in rows if "MLP" in r[0])
    attn = sum(r[1] for r in rows if "Attn" in r[0])
    other = sum(r[1] for r in rows if "Other" in r[0])
    
    print("\nAGGREGATED:")
    print(f"  MLP linears:       {mlp:6.2f} ms ({mlp/total_ms*100:5.1f}%)")
    print(f"  Attention linears: {attn:6.2f} ms ({attn/total_ms*100:5.1f}%)")
    print(f"  Other linears:     {other:6.2f} ms ({other/total_ms*100:5.1f}%)")
    print(f"  Non-linear ops:    {unaccounted:6.2f} ms ({unaccounted_pct:5.1f}%)")
    
    print("\nSPEEDUP ANALYSIS:")
    mlp_pct = mlp / total_ms
    if mlp_pct < 0.25:
        print("  ⚠️  MLP <25% - must sparsify attention for wins")
    elif mlp_pct < 0.35:
        print("  ⚠️  MLP 25-35% - marginal without Q/K/V/O")
    else:
        print("  ✅ MLP >35% - excellent MLP target!")
    
    # Ceiling
    speedup = 1.3
    if mlp > 0:
        new_mlp = mlp / speedup
        new_total = new_mlp + attn + other + unaccounted
        ceiling_mlp = total_ms / new_total
        print(f"\n  If MLP gets {speedup:.1f}× speedup: {ceiling_mlp:.2f}× ceiling")
    
    if mlp > 0 and attn > 0:
        new_both = (mlp + attn) / speedup
        new_total = new_both + other + unaccounted
        ceiling_both = total_ms / new_total
        print(f"  If MLP+Attn get {speedup:.1f}× speedup: {ceiling_both:.2f}× ceiling")
    
    print("="*70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="Explain quantum computing in detail.")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    
    AutoModelForCausalLM, AutoTokenizer = _maybe_import_transformers()
    
    print("="*70)
    print("SparseFlow Prefill Profiler (Per-Call Events)")
    print("="*70)
    print(f"Model:  {args.model}")
    print(f"Batch:  {args.batch}")
    print(f"MaxLen: {args.max_len}")
    print(f"M:      ~{args.batch * args.max_len}")
    print("="*70)
    
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16
    ).to("cuda").eval()
    
    print("Attaching per-call event hooks...")
    handles = attach_hooks(model)
    print(f"  Attached hooks to {len(handles)//2} Linear layers")
    
    # Prepare inputs
    device = next(model.parameters()).device
    inputs = tokenizer([args.prompt] * args.batch, return_tensors="pt",
                      padding=True, truncation=True, max_length=args.max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"\nRunning {args.iters} iterations (warmup={args.warmup})...")
    total_ms, stats_ms, stats_calls = run_profile(model, inputs, args.iters, args.warmup)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    print_breakdown(total_ms, stats_ms, stats_calls)


if __name__ == "__main__":
    main()
