#!/usr/bin/env python3
import subprocess, re, statistics

def run_once():
    out = subprocess.check_output(["python","tools/bench_mlp_vs_linear.py"], text=True, stderr=subprocess.STDOUT)
    def grab(name):
        m = re.search(rf"{name}:\s+([0-9.]+)ms", out)
        return float(m.group(1)) if m else None
    return grab("Dense"), grab("Per-Linear"), grab("SparseFlowMLP")

N=10
dense=[]; per=[]; mlp=[]
print("Running 10 trials...")
for i in range(N):
    print(f"  Trial {i+1}/10...", end=" ", flush=True)
    d,p,m = run_once()
    dense.append(d); per.append(p); mlp.append(m)
    print(f"Dense={d:.2f}, Per={p:.2f}, MLP={m:.2f}")

def stats(xs):
    xs_sorted=sorted(xs)
    mean=statistics.mean(xs)
    p50=xs_sorted[len(xs)//2]
    p95=xs_sorted[int(0.95*(len(xs)-1))]
    return mean,p50,p95

dm,dp50,dp95=stats(dense)
pm,pp50,pp95=stats(per)
mm,mp50,mp95=stats(mlp)

print("="*70)
print("REPEAT BENCH (N=10)")
print("="*70)
print(f"Dense mean/p50/p95:        {dm:.3f} / {dp50:.3f} / {dp95:.3f} ms")
print(f"Per-Linear mean/p50/p95:   {pm:.3f} / {pp50:.3f} / {pp95:.3f} ms   ({dm/pm:.3f}x)")
print(f"SparseFlowMLP mean/p50/p95:{mm:.3f} / {mp50:.3f} / {mp95:.3f} ms   ({dm/mm:.3f}x)")
print("="*70)
print(f"\nFINAL SPEEDUP: {dm/mm:.3f}× (mean over 10 runs)")
if dm/mm > 1.10:
    print("✅ STABLE WIN (>10% speedup)")
elif dm/mm > 1.05:
    print("⚠️  MODEST WIN (5-10% speedup)")
else:
    print("❌ UNSTABLE (within noise)")
print("="*70)
