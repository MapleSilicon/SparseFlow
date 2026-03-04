#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

def predicted_speedup(results, min_sp: float) -> float:
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

        use_sparse = False
        if (not skipped) and acc_ok and (sp is not None) and (sparse is not None):
            try:
                use_sparse = float(sp) >= min_sp
            except:
                use_sparse = False

        chosen_sum += float(sparse) if use_sparse else dense

    return (dense_sum / chosen_sum) if chosen_sum > 0 else 1.0

def main():
    if len(sys.argv) < 4:
        print("Usage: auto_best_override.py <run_json> <out_override_json> <start> <end> <step>")
        raise SystemExit(2)

    runp = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    start = float(sys.argv[3])
    end = float(sys.argv[4])
    step = float(sys.argv[5])

    run = json.loads(runp.read_text())
    results = run.get("results", [])
    if not isinstance(results, list):
        raise SystemExit("Unexpected schema: run['results'] is not a list")

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

    # If plateau, choose a stable middle threshold (closest to 1.12)
    top = sorted(rows, key=lambda x: x[1], reverse=True)
    top_s = top[0][1]
    plateau = [t for t,s in rows if abs(s - top_s) < 1e-9]
    if plateau:
        best_t = min(plateau, key=lambda x: abs(x - 1.12))

    # Write override JSON using your existing generator logic (inline)
    overrides = {
        "source_run": str(runp),
        "run_id": run.get("run_id", None),
        "min_speedup": best_t,
        "overrides": {}
    }

    for r in results:
        if not isinstance(r, dict):
            continue
        shape = r.get("shape", None)
        b = r.get("batch", None)
        s = r.get("seq_len", None)
        if shape is None or b is None or s is None:
            continue

        key = f"{shape}|b={int(b)}|s={int(s)}"
        skipped = bool(r.get("skipped", False))
        gate = r.get("shape_gate", "")
        acc_pass = r.get("acc_gate_pass", True)
        acc_err = r.get("acc_max_err_pct", None)
        sp = r.get("speedup", None)

        try:
            spf = float(sp) if sp is not None else None
        except:
            spf = None

        if skipped:
            overrides["overrides"][key] = {"decision": "dense", "reason": f"skipped ({gate})", "speedup": spf, "acc_max_err_pct": acc_err}
        elif acc_pass is False:
            overrides["overrides"][key] = {"decision": "dense", "reason": "acc_gate_fail", "speedup": spf, "acc_max_err_pct": acc_err}
        elif spf is None:
            overrides["overrides"][key] = {"decision": "dense", "reason": "no-speedup", "speedup": None, "acc_max_err_pct": acc_err}
        elif spf >= best_t:
            overrides["overrides"][key] = {"decision": "sparse", "reason": f"speedup={spf:.3f}>=min", "speedup": spf, "acc_max_err_pct": acc_err}
        else:
            overrides["overrides"][key] = {"decision": "dense", "reason": f"speedup={spf:.3f}<min", "speedup": spf, "acc_max_err_pct": acc_err}

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(overrides, indent=2))

    print(f"Best min_speedup: {best_t:.2f}  predicted overall speedup: {best_s:.4f}x")
    print(f"Wrote: {outp}")

if __name__ == "__main__":
    main()
