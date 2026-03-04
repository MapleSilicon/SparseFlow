#!/usr/bin/env python3
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
j = json.loads(p.read_text())

def walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)

speedup_keys = {"speedup", "speedup_x", "speedupX", "speedup_ratio"}
gate_keys = {"gate", "gated", "gate_decision", "decision"}

root_type = type(j).__name__
print(f"FILE: {p}")
print(f"ROOT TYPE: {root_type}")

if isinstance(j, dict):
    print("TOP KEYS:", sorted(list(j.keys()))[:80])
elif isinstance(j, list):
    print("ROOT LIST LEN:", len(j))

dicts = list(walk(j))
print("DICT COUNT:", len(dicts))

# Find candidate dicts that mention speedup keys
cands = []
for d in dicts:
    if any(k in d for k in speedup_keys) or any(k in d for k in gate_keys):
        cands.append(d)

print("CANDIDATE DICTS (speedup/gate present):", len(cands))

def short(d):
    ks = sorted(d.keys())
    out = {}
    for k in ks:
        v = d[k]
        if k in speedup_keys or k in gate_keys or k in ("op","name","layer","batch","b","seq","s","seqlen","status","result","pass","fail","dense","sparse","metrics","cfg","config","case","shape"):
            out[k] = v
    return out

print("\n--- FIRST 5 CANDIDATES (trimmed) ---")
for i, d in enumerate(cands[:5]):
    print(f"\n[{i}] keys={sorted(list(d.keys()))[:40]}")
    sd = short(d)
    print(sd)

# Also search for any dict that has BOTH dense and sparse blocks
ds = [d for d in dicts if isinstance(d.get("dense", None), dict) or isinstance(d.get("sparse", None), dict)]
print("\nDICTs WITH dense/sparse blocks:", len(ds))
if ds:
    d0 = ds[0]
    print("SAMPLE dense/sparse keys:", sorted(list(d0.keys()))[:40])
    print("dense keys:", sorted(list(d0.get("dense", {}).keys()))[:30])
    print("sparse keys:", sorted(list(d0.get("sparse", {}).keys()))[:30])
