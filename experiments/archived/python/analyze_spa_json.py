#!/usr/bin/env python3
import json
from pathlib import Path

json_path = Path("spa_sparsity.json")
if not json_path.exists():
    print("❌ spa_sparsity.json not found. Run ./run_spa_v06_demo.sh first.")
    exit(1)

data = json.loads(json_path.read_text())

total_ops = data.get("total_operations", 0)
sparse_ops = data.get("sparse_operations", 0)
print(f"=== SPA JSON Analysis ===")
print(f"Total operations: {total_ops}")
print(f"Sparse operations (with row/col info): {sparse_ops}")
print()

for op in data.get("operations", []):
    name = op.get("name", "<unknown>")
    op_id = op.get("id", -1)

    total_rows = op.get("total_rows")
    total_cols = op.get("total_cols")
    zero_rows = op.get("zero_rows")
    zero_cols = op.get("zero_cols")

    rowmask = op.get("rowmask")
    colmask = op.get("colmask")

    print(f"Op #{op_id} : {name}")
    if rowmask is not None:
        print(f"  rowmask = {rowmask} (row_sparsity_pct={op.get('row_sparsity_pct','?')}%)")
    if colmask is not None:
        print(f"  colmask = {colmask} (col_sparsity_pct={op.get('col_sparsity_pct','?')}%)")

    if total_rows is not None and total_cols is not None:
        # Effective non-zero density assuming rectangular live region
        live_rows = total_rows - (zero_rows or 0)
        live_cols = total_cols - (zero_cols or 0)
        total_elems = total_rows * total_cols
        live_elems = live_rows * live_cols
        density = live_elems / total_elems if total_elems > 0 else 0.0
        sparsity = 1.0 - density
        print(f"  total_rows = {total_rows}, zero_rows = {zero_rows}")
        print(f"  total_cols = {total_cols}, zero_cols = {zero_cols}")
        print(f"  effective density ≈ {density*100:.1f}% (sparsity ≈ {sparsity*100:.1f}%)")
    print()
