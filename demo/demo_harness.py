#!/usr/bin/env python3
"""
SparseFlow Demo Harness
Maple Silicon Inc. — RTX 3090 (SM86) / A100 (SM80)
"""

import sys, json, time, threading, subprocess, argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from sparseflow.dispatch import should_use_sparse, classify_projection, THRESHOLDS, MAX_THRESHOLDS
from sparseflow.nn.sparseflow_linear import prune_24_dense_weight, SparseFlowLinear, make_sparseflow_linear

MAX_ALLOWED_ERROR_PCT = 0.5

# TinyLlama 1.1B shapes: (in_features, out_features, description)
TINYLLAMA_SHAPES = {
    "ffn_gate":   (4096, 11008, "FFN gate projection"),
    "ffn_down":   (11008, 4096, "FFN down projection"),
    "attn_qkv":   (4096, 4096,  "Attention QKV projection"),
    "attn_out":   (4096, 4096,  "Attention output projection"),
}

# ── Hardware detection ────────────────────────────────────────────────────────

def detect_hw():
    assert torch.cuda.is_available(), "No CUDA GPU found."
    p = torch.cuda.get_device_properties(0)
    sm = f"sm{p.major}{p.minor}"
    return {
        "name": p.name,
        "sm": sm,
        "vram_gb": round(p.total_memory / 1024**3, 1),
        "sparse_capable": p.major >= 8,
        "is_primary_target": any(x in p.name for x in ["3090","A100","A6000"]),
    }

# ── Power sampling ────────────────────────────────────────────────────────────

class PowerSampler:
    def __init__(self, interval_ms=100):
        self._iv = interval_ms / 1000
        self._samples = []
        self._running = False
        self._t = None
        self._t0 = self._t1 = None

    def _loop(self):
        while self._running:
            try:
                r = subprocess.run(
                    ["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1)
                self._samples.append(float(r.stdout.strip().split("\n")[0]))
            except: pass
            time.sleep(self._iv)

    def start(self):
        self._samples = []
        self._running = True
        self._t0 = time.perf_counter()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._running = False
        self._t1 = time.perf_counter()
        if self._t: self._t.join(timeout=2)
        elapsed = self._t1 - self._t0
        if len(self._samples) < 2:
            return {"joules": None, "avg_watts": None, "note": "nvidia-smi unavailable"}
        avg_w = sum(self._samples) / len(self._samples)
        return {
            "joules": round(avg_w * elapsed, 5),
            "avg_watts": round(avg_w, 2),
            "n_samples": len(self._samples),
            "elapsed_s": round(elapsed, 3),
        }

# ── Timing ────────────────────────────────────────────────────────────────────

def time_linear(layer, x, warmup=10, iters=50):
    for _ in range(warmup):
        _ = layer(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); _ = layer(x); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times)//2], times[int(len(times)*0.95)]

# ── Single config ─────────────────────────────────────────────────────────────

def run_config(batch, seq_len, shape_name, in_f, out_f, hw, run_energy):
    M = batch * seq_len
    gate_ok = should_use_sparse(M, in_f, out_f)
    proj_type = classify_projection(in_f, out_f)
    threshold = THRESHOLDS[proj_type]

    gate = {
        "recommendation": "sparse" if gate_ok else "dense",
        "proj_type": proj_type,
        "threshold": threshold,
        "M_eff": M,
        "reason": (
            f"M_eff={M} >= threshold {threshold} — sparse expected to help"
            if gate_ok else
            (
                f"M_eff={M} > upper bound {MAX_THRESHOLDS[proj_type]} for {proj_type} — bandwidth-saturated, dense wins"
                if M > MAX_THRESHOLDS[proj_type] else
                f"M_eff={M} < lower threshold {threshold} for {proj_type} — overhead not justified"
            )
        ),
    }

    dtype = torch.float16
    torch.manual_seed(42)
    x = torch.randn(M, in_f, dtype=dtype, device="cuda")

    # Dense baseline
    dense_layer = nn.Linear(in_f, out_f, bias=False, dtype=dtype).cuda()
    dense_p50, dense_p95 = time_linear(dense_layer, x)
    dense_toks = round(M / (dense_p50 / 1000), 1)

    # Save fp16 weight BEFORE float() cast mutates it
    weight_fp16 = dense_layer.weight.data.clone().half()

    # Dense reference output for accuracy check (fp16, then cast for diff)
    with torch.no_grad():
        C_dense_fp16 = dense_layer(x)

    # Dense energy
    energy_dense = joules_dense = None
    if run_energy:
        ps = PowerSampler(); ps.start()
        for _ in range(100): _ = dense_layer(x)
        torch.cuda.synchronize()
        energy_dense = ps.stop()
        if energy_dense["joules"]:
            joules_dense = energy_dense["joules"] / (100 * M)

    # Sparse path
    sparse_p50 = sparse_p95 = sparse_toks = None
    speedup = acc_err = acc_ok = None
    energy_sparse = joules_sparse = energy_red = None
    skip_reason = None

    if not hw["sparse_capable"]:
        skip_reason = f"GPU {hw['name']} does not support 2:4 sparse tensor cores (requires SM80+)"
    elif not gate_ok:
        skip_reason = gate["reason"]
    else:
        try:
            # Build a fresh fp16 nn.Linear from the saved fp16 weight,
            # then pass it directly to make_sparseflow_linear
            import torch.nn as nn_inner
            ref_linear = nn_inner.Linear(in_f, out_f, bias=False).cuda()
            # Prune explicitly so ref_linear.weight.data IS the pruned weight
            pruned_w = prune_24_dense_weight(weight_fp16.cuda())
            ref_linear.weight.data = pruned_w
            sparse_layer = make_sparseflow_linear(ref_linear)
            sparse_layer = sparse_layer.cuda()

            sparse_p50, sparse_p95 = time_linear(sparse_layer, x)
            sparse_toks = round(M / (sparse_p50 / 1000), 1)
            speedup = round(dense_p50 / sparse_p50, 3)

            # Accuracy — compare sparse kernel vs a dense matmul using the
            # SAME pruned weight. This tests kernel correctness, not sparsity error.
            with torch.no_grad():
                C_sparse_fp16 = sparse_layer(x)
                # Dense reference: pruned weight via regular matmul
                C_pruned_dense = torch.nn.functional.linear(x, ref_linear.weight.data)
            diff = (C_pruned_dense.float() - C_sparse_fp16.float()).abs()
            max_err = float(diff.max() / (C_pruned_dense.float().abs().max() + 1e-8)) * 100
            acc_err = round(max_err, 4)
            acc_ok = acc_err <= MAX_ALLOWED_ERROR_PCT

            # Sparse energy
            if run_energy:
                ps2 = PowerSampler(); ps2.start()
                for _ in range(100): _ = sparse_layer(x)
                torch.cuda.synchronize()
                energy_sparse = ps2.stop()
                if energy_sparse["joules"] and joules_dense:
                    joules_sparse = energy_sparse["joules"] / (100 * M)
                    energy_red = round((1 - joules_sparse / joules_dense) * 100, 2)

        except Exception as e:
            skip_reason = f"Sparse run error: {e}"

    torch.cuda.empty_cache()
    return {
        "batch": batch, "seq_len": seq_len, "shape": shape_name,
        "in_features": in_f, "out_features": out_f,
        "shape_gate": gate,
        "dense_p50_ms": round(dense_p50, 3),
        "dense_p95_ms": round(dense_p95, 3),
        "dense_toks_per_s": dense_toks,
        "sparse_p50_ms": round(sparse_p50, 3) if sparse_p50 else None,
        "sparse_p95_ms": round(sparse_p95, 3) if sparse_p95 else None,
        "sparse_toks_per_s": sparse_toks,
        "speedup": speedup,
        "acc_max_err_pct": acc_err,
        "acc_gate_pass": acc_ok,
        "joules_per_token_dense": round(joules_dense, 9) if joules_dense else None,
        "joules_per_token_sparse": round(joules_sparse, 9) if joules_sparse else None,
        "energy_reduction_pct": energy_red,
        "skipped": skip_reason,
    }

# ── Report ────────────────────────────────────────────────────────────────────

def print_summary(results, hw):
    print("\n" + "="*65)
    print(f"  SparseFlow Demo — {hw['name']} ({hw['sm']})")
    print("="*65)
    for r in results:
        gate = r["shape_gate"]
        icon = "✅" if gate["recommendation"]=="sparse" else "❌"
        print(f"\n  [{r['shape']}] batch={r['batch']} seq={r['seq_len']}  gate:{icon}")
        print(f"  Dense  p50:{r['dense_p50_ms']:>8.2f}ms  p95:{r['dense_p95_ms']:>8.2f}ms  {r['dense_toks_per_s']:>10.0f} tok/s")
        if r["sparse_p50_ms"]:
            print(f"  Sparse p50:{r['sparse_p50_ms']:>8.2f}ms  p95:{r['sparse_p95_ms']:>8.2f}ms  {r['sparse_toks_per_s']:>10.0f} tok/s")
            print(f"  Speedup: {r['speedup']}×   Max error: {r['acc_max_err_pct']}%  {'PASS ✅' if r['acc_gate_pass'] else 'FAIL ❌'}")
            if r["energy_reduction_pct"] is not None:
                print(f"  Energy: {r['energy_reduction_pct']}% reduction ({r['joules_per_token_dense']:.7f} → {r['joules_per_token_sparse']:.7f} J/tok)")
        elif r["skipped"]:
            print(f"  → {r['skipped']}")
    print("\n" + "="*65)
    wins = [r for r in results if r["speedup"] and r["speedup"] > 1.05]
    if wins:
        avg = sum(r["speedup"] for r in wins) / len(wins)
        print(f"\n  ✅ Winning configs: {len(wins)}/{len(results)}")
        print(f"  ✅ Avg speedup:     {round(avg,3)}×")
        best = max(wins, key=lambda r: r["speedup"])
        print(f"  ✅ Best speedup:    {best['speedup']}× ({best['shape']} batch={best['batch']} seq={best['seq_len']})")
    else:
        print("\n  ℹ️  No speedup at these batch sizes — expected.")
        print("  ℹ️  Run with --batches 512,1024,2048 for production-scale results.")

    all_acc = [r for r in results if r["acc_gate_pass"] is not None]
    if all_acc:
        passed = sum(r["acc_gate_pass"] for r in all_acc)
        print(f"  {'✅' if passed==len(all_acc) else '❌'} Accuracy gate: {passed}/{len(all_acc)} configs ≤ {MAX_ALLOWED_ERROR_PCT}%")
    print()

def write_markdown(results, hw, model, run_id, path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# SparseFlow Benchmark Report",
        f"",
        f"**Run ID:** `{run_id}`  ",
        f"**Date:** {ts}  ",
        f"**GPU:** {hw['name']} ({hw['sm']}, {hw['vram_gb']}GB)  ",
        f"**Model:** {model}  ",
        f"**PyTorch:** {torch.__version__} | **CUDA:** {torch.version.cuda}  ",
        f"",
        f"---",
        f"",
        f"## Results",
        f"",
        f"| Shape | Batch | Seq | Gate | Dense p50 | Sparse p50 | Speedup | Energy ↓ | Acc Error | Acc Gate |",
        f"|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        g = "✅" if r["shape_gate"]["recommendation"]=="sparse" else "❌"
        sp = f"{r['sparse_p50_ms']}ms" if r["sparse_p50_ms"] else "—"
        su = f"**{r['speedup']}×**" if r["speedup"] and r["speedup"]>1.05 else (f"{r['speedup']}×" if r["speedup"] else "—")
        er = f"{r['energy_reduction_pct']}%" if r["energy_reduction_pct"] is not None else "—"
        ae = f"{r['acc_max_err_pct']}%" if r["acc_max_err_pct"] is not None else "—"
        ag = ("✅" if r["acc_gate_pass"] else ("❌" if r["acc_gate_pass"] is False else "—"))
        lines.append(f"| {r['shape']} | {r['batch']} | {r['seq_len']} | {g} | {r['dense_p50_ms']}ms | {sp} | {su} | {er} | {ae} | {ag} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Shape-Gate Thresholds",
        f"",
        f"| Projection Type | Min M_eff |",
        f"|---|---|",
    ]
    for k, v in THRESHOLDS.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Reproduce",
        f"",
        f"```bash",
        f"git clone https://github.com/MapleSilicon/SparseFlow.git",
        f"cd SparseFlow",
        f"python demo/demo_harness.py --run-id {run_id}",
        f"```",
        f"",
        f"_Maple Silicon Inc. — Generated by demo_harness.py_",
    ]
    Path(path).write_text("\n".join(lines))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama")
    p.add_argument("--batches", default="1,4,8,512")
    p.add_argument("--seq-lens", default="128,512")
    p.add_argument("--shapes", default="ffn_gate,attn_qkv")
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    batches = [int(x) for x in args.batches.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    shapes = [s.strip() for s in args.shapes.split(",")]
    run_energy = not args.no_energy

    out_dir = REPO_ROOT / "demo" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    hw = detect_hw()
    print(f"\n  SparseFlow Demo Harness | Run ID: {run_id}")
    print(f"  GPU: {hw['name']} ({hw['sm']}) | VRAM: {hw['vram_gb']}GB")
    print(f"  Sparse capable: {hw['sparse_capable']} | Primary target: {hw['is_primary_target']}")
    print(f"  Shape-gate thresholds: {THRESHOLDS}")
    print(f"  Energy sampling: {'yes' if run_energy else 'no'}")
    print()

    results = []
    total = len(batches) * len(seq_lens) * len(shapes)
    done = 0
    for sname in shapes:
        if sname not in TINYLLAMA_SHAPES:
            print(f"  [WARN] Unknown shape '{sname}', skipping")
            continue
        in_f, out_f, desc = TINYLLAMA_SHAPES[sname]
        for batch in batches:
            for seq_len in seq_lens:
                done += 1
                print(f"  [{done}/{total}] {sname} batch={batch} seq={seq_len} ...", end="", flush=True)
                r = run_config(batch, seq_len, sname, in_f, out_f, hw, run_energy)
                results.append(r)
                status = (f"  {r['speedup']}×" if r["speedup"] and r["speedup"]>1.0
                          else "  gate:skip" if r["skipped"] else "  parity")
                print(status)

    print_summary(results, hw)

    json_path = out_dir / f"run_{run_id}.json"
    md_path = out_dir / f"run_{run_id}.md"
    json_path.write_text(json.dumps({"run_id":run_id,"hardware":hw,"results":results}, indent=2))
    write_markdown(results, hw, args.model, run_id, md_path)
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}\n")
    print(f"  cat {md_path}")

if __name__ == "__main__":
    main()
