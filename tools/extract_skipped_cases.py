#!/usr/bin/env python3
import json, sys
from pathlib import Path

runp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo/results/run_20260304_160101.json")
outp = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("demo/config/skipped_cases_sm89.json")

j = json.loads(runp.read_text())
results = j.get("results", [])
if not isinstance(results, list):
    raise SystemExit("run['results'] missing or not a list")

skipped = []
for r in results:
    if not isinstance(r, dict):
        continue
    if not bool(r.get("skipped", False)):
        continue
    shape = r.get("shape")
    b = r.get("batch")
    s = r.get("seq_len")
    gate = r.get("shape_gate", {})
    if shape is None or b is None or s is None:
        continue
    rec = {
        "shape": str(shape),
        "batch": int(b),
        "seq_len": int(s),
        "M_eff": int(gate.get("M_eff", int(b)*int(s))),
        "proj_type": gate.get("proj_type", None),
        "threshold": gate.get("threshold", None),
        "reason": gate.get("reason", None),
        "recommendation": gate.get("recommendation", None),
    }
    skipped.append(rec)

outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps({"source_run": str(runp), "skipped": skipped}, indent=2))
print(f"Wrote: {outp}  (n={len(skipped)})")
