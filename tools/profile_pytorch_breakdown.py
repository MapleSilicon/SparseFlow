#!/usr/bin/env python3
"""
Use PyTorch profiler to identify "Other ops" bottleneck
"""

import torch
from torch.profiler import profile, ProfilerActivity


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("="*70)
    print("PyTorch Profiler - Identifying 'Other Ops'")
    print("="*70)
    print(f"Model: {model_id}")
    print(f"Batch: 4, MaxLen: 256")
    print("="*70)
    
    print("\nLoading model...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    
    batch = 4
    max_len = 256
    prompt = "Explain sparse tensor cores in simple terms."
    
    inputs = tok([prompt] * batch, return_tensors="pt", padding=True,
                 truncation=True, max_length=max_len).to("cuda")
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs, use_cache=False)
    torch.cuda.synchronize()
    
    # Profile
    print("Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof:
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
    torch.cuda.synchronize()
    
    print("\n" + "="*70)
    print("TOP 30 CUDA OPERATIONS (by total time)")
    print("="*70)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("\nLook for these patterns:")
    print("  - scaled_dot_product_attention / sdpa / flash_attn")
    print("  - layer_norm / rms_norm")
    print("  - aten::linear (your GEMMs)")
    print("  - aten::add, aten::mul, aten::silu (elementwise)")
    print("  - transpose/reshape/copy operations")
    print("="*70)


if __name__ == "__main__":
    main()
