#!/usr/bin/env python3
import os
import sys

required_dirs = [
    'src/sparseflow', 'kernels/cuda', 'kernels/ptx',
    'benchmarks', 'tests/unit', 'tests/integration',
    'tools', 'docs', 'experiments/archived'
]

print("Checking structure...")
all_good = True
for d in required_dirs:
    if os.path.isdir(d):
        print(f"  ✓ {d}")
    else:
        print(f"  ✗ {d}")
        all_good = False

if all_good:
    print("\n✅ Structure verified!")
else:
    print("\n⚠️  Some directories missing")
    sys.exit(1)
