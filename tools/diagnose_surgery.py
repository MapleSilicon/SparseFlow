#!/usr/bin/env python3
"""Diagnose why sparse might be slow"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompts = [
    "Explain quantum computing, machine learning, and sparse tensor cores in detail.",
    "Explain quantum computing. " * 50,  # Medium
    "Explain quantum computing. " * 200,  # Large
]

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

print("Prompt Length Analysis:")
print("="*70)
for i, prompt in enumerate(prompts, 1):
    tokens = tokenizer(prompt, return_tensors="pt")
    seq_len = tokens['input_ids'].shape[1]
    
    print(f"\nPrompt {i}:")
    print(f"  Sequence length: {seq_len} tokens")
    print(f"  Effective M (batch=1): {seq_len}")
    
    if seq_len >= 512:
        print(f"  ✅ Will use SPARSE (M >= 512)")
    else:
        print(f"  ❌ Will use DENSE (M < 512)")
        print(f"     Need {512 - seq_len} more tokens for sparse")

print("\n" + "="*70)
print("Recommendation: Use prompts with >512 tokens for speedup")
