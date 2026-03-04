#!/usr/bin/env python3
import json, sys
from pathlib import Path
from collections import defaultdict

runp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo/results/run_20260304_160101.json")
j = json.loads(runp.read_text())
results = j.get("results", [])
if not isinstance(results, list):
    raise SystemExit("run['results'] missing or not a list")

sk = defaultdict(list)
for r in results:
    if not isinstance(r, dict):
        continue
    if not bool(r.get("skipped", False)):
        continue
    shape = r.get("shape", "unknown")
    b = int(r.get("batch", -1))
    s = int(r.get("seq_len", -1))
    gate = r.get("shape_gate", "")
    m_eff = b * s if b > 0 and s > 0 else -1
    sk[shape].append((m_eff, b, s, gate))

print(f"RUN: {runp}")
total = 0
for shape in sorted(sk.keys()):
    rows = sorted(sk[shape], key=lambda x: x[0])
    total += len(rows)
    print(f"\n[{shape}] skipped={len(rows)}")
    for m_eff,b,s,gate in rows:
        print(f"  M_eff={m_eff:<7d}  b={b:<4d} s={s:<4d}  gate={gate}")
print(f"\nTotal skipped cases: {total}")
