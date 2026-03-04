#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

def main():
    if len(sys.argv) < 3:
        print("Usage: eval_override_impact.py <run_json> <override_json>")
        raise SystemExit(2)

    runp = Path(sys.argv[1])
    ovp  = Path(sys.argv[2])

    run = json.loads(runp.read_text())
    ov  = json.loads(ovp.read_text())

    results = run.get("results", [])
    overrides = ov.get("overrides", {})

    if not isinstance(results, list) or not isinstance(overrides, dict):
        raise SystemExit("Unexpected schema in run or overrides json.")

    tot_dense_ms = 0.0
    tot_chosen_ms = 0.0

    # per-shape sums
    s_dense = defaultdict(float)
    s_chosen = defaultdict(float)
    counts = defaultdict(int)
    decisions = defaultdict(int)

    missing = 0

    for r in results:
        if not isinstance(r, dict):
            continue
        shape = r.get("shape")
        b = r.get("batch")
        s = r.get("seq_len")
        if shape is None or b is None or s is None:
            continue

        key = f"{shape}|b={int(b)}|s={int(s)}"
        rec = overrides.get(key, None)
        if rec is None:
            # if no override exists, default dense (safe)
            decision = "dense"
        else:
            decision = rec.get("decision", "dense")

        dense_ms = r.get("dense_p50_ms", None)
        sparse_ms = r.get("sparse_p50_ms", None)

        if dense_ms is None:
            missing += 1
            continue

        dense_ms = float(dense_ms)
        chosen_ms = dense_ms

        # Use sparse time only if decision==sparse and sparse_ms exists and not skipped
        skipped = bool(r.get("skipped", False))
        if (decision == "sparse") and (not skipped) and (sparse_ms is not None):
            chosen_ms = float(sparse_ms)

        tot_dense_ms += dense_ms
        tot_chosen_ms += chosen_ms

        s_dense[shape] += dense_ms
        s_chosen[shape] += chosen_ms
        counts[shape] += 1
        decisions[decision] += 1

    def speedup(d, c):
        return (d / c) if c > 0 else 1.0

    overall = speedup(tot_dense_ms, tot_chosen_ms)

    print(f"RUN:      {runp}")
    print(f"OVERRIDES:{ovp}")
    print(f"Decisions: sparse={decisions['sparse']}  dense={decisions['dense']}")
    print(f"Missing dense_p50_ms rows skipped: {missing}")
    print("")
    print(f"Overall predicted speedup vs all-dense (sum p50 ms): {overall:.4f}x")
    print(f"  sum_dense_ms =  {tot_dense_ms:.3f} ms")
    print(f"  sum_chosen_ms = {tot_chosen_ms:.3f} ms")
    print("")
    print("Per-shape predicted speedup:")
    for shape in sorted(s_dense.keys()):
        sp = speedup(s_dense[shape], s_chosen[shape])
        print(f"  {shape:10s}  n={counts[shape]:3d}  speedup={sp:.4f}x   dense_ms={s_dense[shape]:.3f}  chosen_ms={s_chosen[shape]:.3f}")

if __name__ == "__main__":
    main()
