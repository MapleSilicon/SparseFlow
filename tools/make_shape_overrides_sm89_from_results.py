#!/usr/bin/env python3
"""
Generate per-shape overrides from SparseFlow demo run JSON.

Expected schema:
  root: { hardware: {...}, run_id: "...", results: [ {...}, ... ] }

Each result item typically includes:
  shape: str            (e.g. "ffn_gate", "ffn_down", "attn_qkv")
  batch: int
  seq_len: int
  speedup: float
  skipped: bool
  shape_gate: str       (reason/metadata)
  acc_gate_pass: bool
  acc_max_err_pct: float (optional)

Decision rule:
  - skipped == True -> dense
  - acc_gate_pass == False -> dense
  - else if speedup >= min_speedup -> sparse
  - else -> dense
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, Tuple

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
    results = j.get("results", None)
    if not isinstance(results, list):
        raise SystemExit("Unexpected schema: root['results'] is missing or not a list")

    # Dedup by (shape,batch,seq_len): keep best non-skipped, else best speedup.
    best: Dict[Tuple[str,int,int], Dict[str, Any]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        shape = r.get("shape", None)
        b = r.get("batch", None)
        s = r.get("seq_len", None)
        if shape is None or b is None or s is None:
            continue
        try:
            key = (str(shape), int(b), int(s))
        except:
            continue

        # Prefer entries that are not skipped, then higher speedup
        prev = best.get(key)
        cur_sk = bool(r.get("skipped", False))
        cur_sp = r.get("speedup", None)
        try:
            cur_sp = float(cur_sp) if cur_sp is not None else None
        except:
            cur_sp = None

        if prev is None:
            best[key] = r
            continue

        prev_sk = bool(prev.get("skipped", False))
        prev_sp = prev.get("speedup", None)
        try:
            prev_sp = float(prev_sp) if prev_sp is not None else None
        except:
            prev_sp = None

        if prev_sk and not cur_sk:
            best[key] = r
        elif (prev_sk == cur_sk) and (prev_sp is None) and (cur_sp is not None):
            best[key] = r
        elif (prev_sk == cur_sk) and (prev_sp is not None) and (cur_sp is not None) and (cur_sp > prev_sp):
            best[key] = r

    if not best:
        raise SystemExit("No valid (shape,batch,seq_len) entries found in results[]")

    out = {
        "source_run": str(runp),
        "run_id": j.get("run_id", None),
        "min_speedup": args.min_speedup,
        "overrides": {}
    }

    n_sparse = n_dense = 0
    for (shape,b,s), r in sorted(best.items()):
        sp = r.get("speedup", None)
        try:
            spf = float(sp) if sp is not None else None
        except:
            spf = None

        skipped = bool(r.get("skipped", False))
        gate = r.get("shape_gate", "")
        acc_pass = r.get("acc_gate_pass", True)
        acc_err = r.get("acc_max_err_pct", None)

        k = f"{shape}|b={b}|s={s}"

        if skipped:
            out["overrides"][k] = {"decision": "dense", "reason": f"skipped ({gate})", "speedup": spf, "acc_max_err_pct": acc_err}
            n_dense += 1
        elif acc_pass is False:
            out["overrides"][k] = {"decision": "dense", "reason": "acc_gate_fail", "speedup": spf, "acc_max_err_pct": acc_err}
            n_dense += 1
        elif spf is None:
            out["overrides"][k] = {"decision": "dense", "reason": "no-speedup", "speedup": None, "acc_max_err_pct": acc_err}
            n_dense += 1
        elif spf >= args.min_speedup:
            out["overrides"][k] = {"decision": "sparse", "reason": f"speedup={spf:.3f}>=min", "speedup": spf, "acc_max_err_pct": acc_err}
            n_sparse += 1
        else:
            out["overrides"][k] = {"decision": "dense", "reason": f"speedup={spf:.3f}<min", "speedup": spf, "acc_max_err_pct": acc_err}
            n_dense += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"Wrote: {args.out}")
    print(f"Shapes: {len(out['overrides'])}  sparse={n_sparse}  dense={n_dense}")

if __name__ == "__main__":
    main()
