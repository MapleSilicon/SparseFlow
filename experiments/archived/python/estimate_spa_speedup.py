#!/usr/bin/env python3
import json
from pathlib import Path

json_path = Path("spa_sparsity.json")
if not json_path.exists():
    print("❌ spa_sparsity.json not found. Run ./run_spa_v06_demo.sh first.")
    exit(1)

data = json.loads(json_path.read_text())

matmul_ops = [op for op in data.get("operations", []) if op.get("name") == "linalg.matmul"]
if not matmul_ops:
    print("❌ No linalg.matmul op found in spa_sparsity.json")
    exit(1)

op = matmul_ops[0]  # assume single matmul for this demo
total_rows = op.get("total_rows")
total_cols = op.get("total_cols")
zero_rows = op.get("zero_rows") or 0
zero_cols = op.get("zero_cols") or 0

if total_rows is None or total_cols is None:
    print("❌ Missing total_rows/total_cols in JSON for linalg.matmul")
    exit(1)

live_rows = total_rows - zero_rows
live_cols = total_cols - zero_cols

total_elems = total_rows * total_cols
live_elems = live_rows * live_cols

if total_elems == 0:
    print("❌ total_elems is zero, cannot compute density")
    exit(1)

density = live_elems / total_elems
sparsity = 1.0 - density

# Relative MAC count / speedup:
# Dense: 1.0 "unit" of work
# SPA-guided: density * 1.0 (same K, same scaling, just normalized)
# So ideal speedup = 1 / density
if density == 0:
    print("=== SPA Speedup Estimate ===")
    print("All rows/cols are zeroed → theoretical infinite speedup (trivial kernel).")
else:
    speedup = 1.0 / density
    print("=== SPA Speedup Estimate (linalg.matmul) ===")
    print(f"Total rows     : {total_rows} (zero_rows = {zero_rows})")
    print(f"Total cols     : {total_cols} (zero_cols = {zero_cols})")
    print(f"Effective density : {density*100:.1f}%")
    print(f"Effective sparsity: {sparsity*100:.1f}%")
    print(f"Ideal relative speedup (if hardware/runtime fully exploits SPA masks): ~{speedup:.2f}x")
