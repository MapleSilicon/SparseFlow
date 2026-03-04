#!/usr/bin/env python3
import json, sys, re
from pathlib import Path

runp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo/results/run_20260304_160101.json")
j = json.loads(runp.read_text())
results = j.get("results", [])
if not isinstance(results, list):
    raise SystemExit("run['results'] missing or not a list")

bad = []
for r in results:
    if not isinstance(r, dict):
        continue
    if not bool(r.get("skipped", False)):
        continue
    gate = r.get("shape_gate", None)
    if not isinstance(gate, dict):
        continue
    m_eff = gate.get("M_eff", None)
    thr = gate.get("threshold", None)
    reason = gate.get("reason", "")
    if m_eff is None or thr is None or not isinstance(reason, str):
        continue

    # Detect the specific wrong pattern: "M_eff=... < threshold ..."
    if "< threshold" in reason and int(m_eff) >= int(thr):
        bad.append({
            "shape": r.get("shape"),
            "batch": r.get("batch"),
            "seq_len": r.get("seq_len"),
            "M_eff": int(m_eff),
            "threshold": int(thr),
            "reason": reason,
            "proj_type": gate.get("proj_type"),
            "recommendation": gate.get("recommendation"),
        })

print(f"RUN: {runp}")
print(f"Skipped cases: {sum(1 for rr in results if isinstance(rr, dict) and rr.get('skipped'))}")
print(f"Inconsistent '< threshold' reasons: {len(bad)}")
for x in bad:
    print(f"  {x['shape']} b={x['batch']} s={x['seq_len']}  M_eff={x['M_eff']} thr={x['threshold']}  proj={x['proj_type']}")
    print(f"    reason: {x['reason']}")
