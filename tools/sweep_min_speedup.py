#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

def predicted_speedup(results, min_sp):
    dense_sum = 0.0
    chosen_sum = 0.0

    for r in results:
        if not isinstance(r, dict):
            continue
        dense = r.get("dense_p50_ms", None)
        sparse = r.get("sparse_p50_ms", None)
        sp = r.get("speedup", None)
        skipped = bool(r.get("skipped", False))
        acc_ok = r.get("acc_gate_pass", True)

        if dense is None:
            continue
        dense = float(dense)
        dense_sum += dense

        # default decision: dense
        use_sparse = False
        if (not skipped) and acc_ok and (sp is not None) and (sparse is not None):
            try:
                use_sparse = float(sp) >= min_sp
            except:
                use_sparse = False

        chosen_sum += float(sparse) if use_sparse else dense

    if chosen_sum <= 0:
        return 1.0
    return dense_sum / chosen_sum

def main():
    if len(sys.argv) < 2:
        print("Usage: sweep_min_speedup.py <run_json> [start end step]")
        raise SystemExit(2)

    runp = Path(sys.argv[1])
    run = json.loads(runp.read_text())
    results = run.get("results", [])
    if not isinstance(results, list):
        raise SystemExit("Unexpected schema: run['results'] is not a list")

    start = float(sys.argv[2]) if len(sys.argv) > 2 else 1.05
    end   = float(sys.argv[3]) if len(sys.argv) > 3 else 1.35
    step  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01

    best_t = None
    best_s = -1.0

    t = start
    rows = []
    while t <= end + 1e-9:
        s = predicted_speedup(results, t)
        rows.append((t, s))
        if s > best_s:
            best_s = s
            best_t = t
        t += step

    print(f"RUN: {runp}")
    print(f"Best min_speedup = {best_t:.2f}  predicted overall speedup = {best_s:.4f}x")
    print("")
    print("Top 10 thresholds:")
    rows.sort(key=lambda x: x[1], reverse=True)
    for t,s in rows[:10]:
        print(f"  min={t:.2f}  speedup={s:.4f}x")

if __name__ == "__main__":
    main()
