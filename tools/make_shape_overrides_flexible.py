#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

SPEEDUP_KEYS = ("speedup", "speedup_x", "speedupX", "speedup_ratio")
OP_KEYS      = ("op", "name", "layer", "kind")
BATCH_KEYS   = ("batch", "b")
SEQ_KEYS     = ("seq", "s", "seqlen", "t")

CONTAINERS   = ("cfg", "config", "case", "shape", "params", "args", "meta", "spec")

def walk(x: Any):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)

def get_first(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None

def get_from_containers(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    # first try direct
    v = get_first(d, keys)
    if v is not None:
        return v
    # then try known one-level nested containers
    for c in CONTAINERS:
        sub = d.get(c, None)
        if isinstance(sub, dict):
            v = get_first(sub, keys)
            if v is not None:
                return v
    return None

def parse_status(d: Dict[str, Any]) -> str:
    s = d.get("status", d.get("result", ""))
    if isinstance(s, str):
        su = s.upper()
        if "PASS" in su: return "PASS"
        if "FAIL" in su: return "FAIL"
    if d.get("pass", None) is True: return "PASS"
    if d.get("fail", None) is True: return "FAIL"
    return ""

def parse_gate(d: Dict[str, Any]) -> Optional[str]:
    g = d.get("gate", d.get("gated", d.get("gate_decision", d.get("decision", None))))
    if g is None:
        # maybe nested
        for c in CONTAINERS:
            sub = d.get(c, None)
            if isinstance(sub, dict):
                g = sub.get("gate", sub.get("decision", None))
                if g is not None:
                    break
    if isinstance(g, str):
        return g.lower()
    if isinstance(g, bool):
        return "run" if g else "skip"
    return None

def try_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, str):
            # strip unicode × or trailing x
            s = x.replace("×","").replace("x","").strip()
            return float(s)
    except:
        return None
    return None

def extract_speedup(d: Dict[str, Any]) -> Optional[float]:
    # direct keys
    for k in SPEEDUP_KEYS:
        if k in d:
            v = try_float(d[k])
            if v is not None:
                return v
    # nested metrics
    m = d.get("metrics", None)
    if isinstance(m, dict):
        for k in SPEEDUP_KEYS:
            if k in m:
                v = try_float(m[k])
                if v is not None:
                    return v
    return None

def extract_p50_ms(block: Dict[str, Any]) -> Optional[float]:
    for k in ("p50_ms", "p50", "p50MS", "median_ms", "median"):
        if k in block:
            v = try_float(block[k])
            if v is not None:
                return v
    return None

def extract_record(d: Dict[str, Any]) -> Optional[Tuple[str,int,int,Optional[float],str,str]]:
    op = get_from_containers(d, OP_KEYS)
    b  = get_from_containers(d, BATCH_KEYS)
    s  = get_from_containers(d, SEQ_KEYS)
    if op is None or b is None or s is None:
        return None

    try:
        op = str(op)
        b = int(b)
        s = int(s)
    except:
        return None

    st = parse_status(d)
    gate = parse_gate(d) or ""

    sp = extract_speedup(d)

    # if speedup absent, try to compute from dense/sparse p50
    if sp is None:
        dense = d.get("dense", None)
        sparse = d.get("sparse", None)
        if isinstance(dense, dict) and isinstance(sparse, dict):
            dp = extract_p50_ms(dense)
            sp50 = extract_p50_ms(sparse)
            if dp is not None and sp50 is not None and sp50 != 0.0:
                sp = dp / sp50

    return (op, b, s, sp, st, gate)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("--min_speedup", type=float, default=1.20)
    ap.add_argument("--out", default="demo/config/shape_overrides_sm89.json")
    args = ap.parse_args()

    runp = Path(args.run_json)
    j = json.loads(runp.read_text())

    best: Dict[Tuple[str,int,int], Tuple[str,int,int,Optional[float],str,str]] = {}

    for d in walk(j):
        if not isinstance(d, dict):
            continue
        rec = extract_record(d)
        if rec is None:
            continue
        op,b,s,sp,st,gate = rec
        k = (op,b,s)

        # Keep the record with the highest speedup (or any speedup over None)
        if k not in best:
            best[k] = rec
        else:
            prev = best[k]
            prev_sp = prev[3]
            if prev_sp is None and sp is not None:
                best[k] = rec
            elif prev_sp is not None and sp is not None and sp > prev_sp:
                best[k] = rec

    if not best:
        raise SystemExit("No per-shape records found. Run peek_run_json.py output and we’ll adapt to your schema.")

    overrides = {
        "source_run": str(runp),
        "min_speedup": args.min_speedup,
        "overrides": {}
    }

    n_sparse = n_dense = 0
    for (op,b,s), rec in sorted(best.items()):
        op,b,s,sp,st,gate = rec
        key = f"{op}|b={b}|s={s}"

        # If explicitly gated/skip, force dense
        if "skip" in gate:
            overrides["overrides"][key] = {"decision": "dense", "reason": f"gate={gate}"}
            n_dense += 1
            continue

        # If FAIL, force dense
        if st == "FAIL":
            overrides["overrides"][key] = {"decision": "dense", "reason": "FAIL"}
            n_dense += 1
            continue

        # If no speedup measured, default dense (safe)
        if sp is None:
            overrides["overrides"][key] = {"decision": "dense", "reason": "no-speedup"}
            n_dense += 1
            continue

        if sp >= args.min_speedup:
            overrides["overrides"][key] = {"decision": "sparse", "reason": f"speedup={sp:.3f}>=min"}
            n_sparse += 1
        else:
            overrides["overrides"][key] = {"decision": "dense", "reason": f"speedup={sp:.3f}<min"}
            n_dense += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(overrides, indent=2))

    print(f"Wrote {args.out}")
    print(f"Shapes: {len(overrides['overrides'])}  sparse={n_sparse}  dense={n_dense}")
