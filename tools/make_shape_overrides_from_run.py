#!/usr/bin/env python3
"""
Create per-shape dense/sparse overrides from a run JSON.

Decision rule:
  - FAIL -> dense
  - PASS + speedup >= min_speedup -> sparse
  - otherwise -> dense

Output:
  demo/config/shape_overrides_sm89.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

def iter_dicts(x: Any):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from iter_dicts(v)
    elif isinstance(x, list):
        for v in x:
            yield from iter_dicts(v)

def get(d: Dict[str, Any], *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default

def norm_status(d: Dict[str, Any]) -> str:
    s = get(d, "status", "result", default="")
    if isinstance(s, str):
        s2 = s.upper()
        if "PASS" in s2: return "PASS"
        if "FAIL" in s2: return "FAIL"
    if get(d, "pass", default=None) is True: return "PASS"
    if get(d, "fail", default=None) is True: return "FAIL"
    return ""

def extract_rows(j: Any) -> List[Tuple[str,int,int,float,str]]:
    rows: List[Tuple[str,int,int,float,str]] = []

    for d in iter_dicts(j):
        if not isinstance(d, dict):
            continue

        op = get(d, "op", "name", "layer", default=None)
        b  = get(d, "batch", "b", default=None)
        s  = get(d, "seq", "seqlen", "t", "s", default=None)
        st = norm_status(d)

        sp = get(d, "speedup", "speedup_x", default=None)
        if op is not None and b is not None and s is not None and sp is not None:
            try:
                rows.append((str(op), int(b), int(s), float(sp), st))
            except:
                pass
            continue

        # fallback: compute speedup from dense/sparse p50 if present
        dense = get(d, "dense", default=None)
        sparse = get(d, "sparse", default=None)
        if op is None or b is None or s is None or not isinstance(dense, dict) or not isinstance(sparse, dict):
            continue

        dp50 = get(dense, "p50_ms", "p50", default=None)
        sp50 = get(sparse, "p50_ms", "p50", default=None)
        if dp50 is None or sp50 is None:
            continue

        try:
            rows.append((str(op), int(b), int(s), float(dp50)/float(sp50), st))
        except:
            pass

    # dedup best speedup per (op,b,s)
    best: Dict[Tuple[str,int,int], Tuple[str,int,int,float,str]] = {}
    for op,b,s,spd,st in rows:
        k = (op,b,s)
        if k not in best or spd > best[k][3]:
            best[k] = (op,b,s,spd,st)

    return list(best.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("--min_speedup", type=float, default=1.20)
    ap.add_argument("--out", default="demo/config/shape_overrides_sm89.json")
    args = ap.parse_args()

    runp = Path(args.run_json)
    if not runp.exists():
        raise SystemExit(f"Run JSON not found: {runp}")

    j = json.loads(runp.read_text())
    rows = extract_rows(j)
    if not rows:
        raise SystemExit("No per-shape rows found in run JSON (structure unexpected).")

    out = {
        "source_run": str(runp),
        "min_speedup": args.min_speedup,
        "overrides": {}
    }

    for op,b,s,spd,st in rows:
        key = f"{op}|b={b}|s={s}"
        if st == "FAIL":
            out["overrides"][key] = {"decision": "dense", "reason": "FAIL"}
        elif spd >= args.min_speedup:
            out["overrides"][key] = {"decision": "sparse", "reason": f"speedup={spd:.3f}>=min"}
        else:
            out["overrides"][key] = {"decision": "dense", "reason": f"speedup={spd:.3f}<min"}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out} ({len(out['overrides'])} shapes)")

if __name__ == "__main__":
    main()
