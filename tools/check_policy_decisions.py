#!/usr/bin/env python3
"""Check which ops use sparse at different M values"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparseflow.nn.policy import SparseFlowPolicy

policy = SparseFlowPolicy()

print("="*70)
print("Policy Decision Matrix")
print("="*70)
print(f"Thresholds: gate/up={policy.min_M_gate_up}, down={policy.min_M_down}")
print("="*70)

ops = ["gate_proj", "up_proj", "down_proj"]
Ms = [128, 256, 384, 512, 768, 1024]

print(f"\n{'M':<8}", end="")
for op in ops:
    print(f"{op:<15}", end="")
print()
print("-"*70)

for M in Ms:
    print(f"{M:<8}", end="")
    for op in ops:
        decision = "SPARSE" if policy.should_use_sparse(op, M) else "DENSE"
        print(f"{decision:<15}", end="")
    print()

print("="*70)
print("\nKey insight:")
print("- gate/up use SPARSE from M=256+ (1.33× speedup)")
print("- down uses DENSE until M=512 (avoids 0.98× slowdown at M=256)")
print("="*70)
