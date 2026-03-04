#!/usr/bin/env python3
import json, sys
from pathlib import Path

base_override = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo/config/shape_overrides_sm89_best.json")
skipped_file  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("demo/config/skipped_cases_sm89.json")
outp          = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("demo/config/shape_overrides_sm89_probe_force_skipped.json")

base = json.loads(base_override.read_text())
sk = json.loads(skipped_file.read_text())

ov = base.get("overrides", {})
if not isinstance(ov, dict):
    raise SystemExit("Base override file missing 'overrides' dict")

items = sk.get("skipped", [])
if not isinstance(items, list):
    raise SystemExit("Skipped file missing 'skipped' list")

forced = 0
for it in items:
    if not isinstance(it, dict):
        continue
    shape = it.get("shape")
    b = it.get("batch")
    s = it.get("seq_len")
    if shape is None or b is None or s is None:
        continue
    key = f"{shape}|b={int(b)}|s={int(s)}"
    ov[key] = {
        "decision": "sparse",
        "reason": "PROBE: force sparse on previously skipped case",
        "speedup": None,
        "acc_max_err_pct": None
    }
    forced += 1

base["overrides"] = ov
base["probe_note"] = "This file forces SPARSE on previously skipped cases to measure true speedups."
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(base, indent=2))
print(f"Wrote: {outp}  (forced_sparse={forced})")
