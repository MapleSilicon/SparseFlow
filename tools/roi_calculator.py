#!/usr/bin/env python3
"""
SparseFlow ROI Calculator (Production-Grade)

Two modes:
1. THROUGHPUT mode (recommended): Use measured dense RPS/GPU from serving
2. LATENCY mode (fallback): Estimate from latency + concurrency

Usage:
  # Throughput mode (recommended)
  python tools/roi_calculator.py throughput \
    --tokens-per-day 1000000000 \
    --tokens-per-request 1024 \
    --dense-rps-per-gpu 3.0 \
    --speedup 1.42 \
    --gpu-hourly-cost 2.50 \
    --efficiency 0.90 \
    --gpu-type "A100 80GB"
"""

import argparse
import math
from dataclasses import dataclass

HOURS_PER_MONTH = 730.0

@dataclass(frozen=True)
class ROIInputs:
    gpu_type: str
    gpu_hourly_cost: float
    tokens_per_day: float
    tokens_per_request: float
    efficiency: float  # Real-world multiplier [0..1]

@dataclass(frozen=True)
class ROIResult:
    mode: str
    speedup: float
    requests_per_day: float
    target_rps_total: float
    dense_rps_per_gpu: float
    sparse_rps_per_gpu: float
    dense_gpus_needed: int
    sparse_gpus_needed: int
    gpus_saved: int
    dense_cost_monthly: float
    sparse_cost_monthly: float
    savings_monthly: float
    savings_yearly: float

def fmt_money(x: float) -> str:
    return f"${x:,.2f}"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / b))

def calc_cost(gpus: int, hourly_cost: float) -> float:
    return float(gpus) * hourly_cost * HOURS_PER_MONTH

def solve_throughput_mode(inp: ROIInputs, dense_rps_per_gpu: float, speedup: float) -> ROIResult:
    requests_per_day = inp.tokens_per_day / inp.tokens_per_request
    target_rps_total = requests_per_day / 86400.0
    
    # Conservative: sparse_rps = dense_rps * (1 + efficiency*(speedup-1))
    sparse_rps_per_gpu = dense_rps_per_gpu * (1.0 + inp.efficiency * (speedup - 1.0))
    
    dense_gpus = ceil_div(target_rps_total, dense_rps_per_gpu)
    sparse_gpus = ceil_div(target_rps_total, sparse_rps_per_gpu)
    gpus_saved = dense_gpus - sparse_gpus
    
    dense_cost = calc_cost(dense_gpus, inp.gpu_hourly_cost)
    sparse_cost = calc_cost(sparse_gpus, inp.gpu_hourly_cost)
    
    return ROIResult(
        mode="throughput",
        speedup=speedup,
        requests_per_day=requests_per_day,
        target_rps_total=target_rps_total,
        dense_rps_per_gpu=dense_rps_per_gpu,
        sparse_rps_per_gpu=sparse_rps_per_gpu,
        dense_gpus_needed=dense_gpus,
        sparse_gpus_needed=sparse_gpus,
        gpus_saved=gpus_saved,
        dense_cost_monthly=dense_cost,
        sparse_cost_monthly=sparse_cost,
        savings_monthly=dense_cost - sparse_cost,
        savings_yearly=(dense_cost - sparse_cost) * 12.0,
    )

def print_report(inp: ROIInputs, res: ROIResult) -> None:
    print("=" * 78)
    print("               SparseFlow ROI Calculator")
    print("=" * 78)
    
    print("\n📊 Workload")
    print(f"  GPU Type:            {inp.gpu_type}")
    print(f"  GPU Hourly Cost:     {fmt_money(inp.gpu_hourly_cost)}/hr")
    print(f"  Tokens/Day:          {inp.tokens_per_day:,.0f}")
    print(f"  Tokens/Request:      {inp.tokens_per_request:,.0f}")
    print(f"  Requests/Day:        {res.requests_per_day:,.0f}")
    print(f"  Target Throughput:   {res.target_rps_total:.4f} RPS")
    
    print("\n⚡ Performance")
    print(f"  Speedup:             {res.speedup:.2f}×")
    print(f"  Efficiency Factor:   {inp.efficiency:.0%} (conservative)")
    print(f"  Dense RPS/GPU:       {res.dense_rps_per_gpu:.4f}")
    print(f"  Sparse RPS/GPU:      {res.sparse_rps_per_gpu:.4f}")
    
    print("\n🖥️  GPU Requirements")
    print(f"  Dense (baseline):    {res.dense_gpus_needed} GPUs")
    print(f"  Sparse (SparseFlow): {res.sparse_gpus_needed} GPUs")
    print(f"  GPUs Saved:          {res.gpus_saved} GPUs")
    
    print("\n💰 Cost Analysis")
    print(f"  Dense Monthly:       {fmt_money(res.dense_cost_monthly)}")
    print(f"  Sparse Monthly:      {fmt_money(res.sparse_cost_monthly)}")
    print(f"  💵 Monthly Savings:  {fmt_money(res.savings_monthly)}")
    print(f"  💵 Yearly Savings:   {fmt_money(res.savings_yearly)}")
    
    if res.gpus_saved > 0:
        roi_percent = (res.savings_monthly / res.dense_cost_monthly) * 100
        print(f"  ROI:                 {roi_percent:.1f}%")
    
    print("\n" + "=" * 78)
    print(f"🎯 Bottom Line: Save {fmt_money(res.savings_yearly)} per year")
    print("=" * 78)
    print()

def main():
    p = argparse.ArgumentParser(description="Calculate SparseFlow ROI")
    sub = p.add_subparsers(dest="cmd", required=True)
    
    # Throughput mode
    p_thr = sub.add_parser("throughput", help="Use measured RPS/GPU (recommended)")
    p_thr.add_argument("--tokens-per-day", type=float, required=True)
    p_thr.add_argument("--tokens-per-request", type=float, required=True)
    p_thr.add_argument("--dense-rps-per-gpu", type=float, required=True,
                       help="Measured dense requests/sec per GPU")
    p_thr.add_argument("--speedup", type=float, required=True,
                       help="SparseFlow speedup (from benchmarks)")
    p_thr.add_argument("--gpu-hourly-cost", type=float, required=True)
    p_thr.add_argument("--gpu-type", default="A100 80GB")
    p_thr.add_argument("--efficiency", type=float, default=0.90,
                       help="Real-world efficiency factor (default 0.90)")
    
    args = p.parse_args()
    
    if args.cmd == "throughput":
        inp = ROIInputs(
            gpu_type=args.gpu_type,
            gpu_hourly_cost=args.gpu_hourly_cost,
            tokens_per_day=args.tokens_per_day,
            tokens_per_request=args.tokens_per_request,
            efficiency=clamp(args.efficiency, 0.0, 1.0),
        )
        
        res = solve_throughput_mode(inp, args.dense_rps_per_gpu, args.speedup)
        print_report(inp, res)

if __name__ == "__main__":
    main()
