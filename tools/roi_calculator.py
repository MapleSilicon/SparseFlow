#!/usr/bin/env python3
"""
SparseFlow ROI Calculator (Production-Grade)

Mode:
  - throughput (recommended): use measured dense RPS/GPU from your serving stack

Why this tool is conservative:
  - speedup gains are discounted by an efficiency factor (default 0.90)
  - outputs are capacity/cost estimates, not guarantees

Example:
  python tools/roi_calculator.py throughput \
    --tokens-per-day 1000000000 \
    --tokens-per-request 1024 \
    --dense-rps-per-gpu 3.0 \
    --speedup 1.42 \
    --gpu-hourly-cost 2.50 \
    --efficiency 0.90 \
    --gpu-type "A100 80GB" \
    --monthly-fee 5000
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

HOURS_PER_MONTH = 730.0
SECONDS_PER_DAY = 86400.0


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
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x:,.2f}"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ceil_div(a: float, b: float) -> int:
    if b <= 0:
        raise ValueError("division by non-positive value")
    return int(math.ceil(a / b))


def calc_cost(gpus: int, hourly_cost: float) -> float:
    return float(gpus) * hourly_cost * HOURS_PER_MONTH


def validate_inputs(tokens_per_day: float, tokens_per_request: float, dense_rps_per_gpu: float, speedup: float, gpu_hourly_cost: float) -> None:
    if tokens_per_day <= 0:
        raise ValueError("--tokens-per-day must be > 0")
    if tokens_per_request <= 0:
        raise ValueError("--tokens-per-request must be > 0")
    if dense_rps_per_gpu <= 0:
        raise ValueError("--dense-rps-per-gpu must be > 0")
    if speedup <= 0:
        raise ValueError("--speedup must be > 0")
    if gpu_hourly_cost <= 0:
        raise ValueError("--gpu-hourly-cost must be > 0")


def solve_throughput_mode(inp: ROIInputs, dense_rps_per_gpu: float, speedup: float) -> ROIResult:
    # Workload
    requests_per_day = inp.tokens_per_day / inp.tokens_per_request
    target_rps_total = requests_per_day / SECONDS_PER_DAY

    # Conservative uplift:
    # sparse_rps = dense_rps * (1 + efficiency*(speedup-1))
    uplift = 1.0 + inp.efficiency * (speedup - 1.0)

    # Clamp uplift to [1, speedup] to avoid overclaiming
    uplift = clamp(uplift, 1.0, speedup)

    sparse_rps_per_gpu = dense_rps_per_gpu * uplift

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


def print_report(inp: ROIInputs, res: ROIResult, monthly_fee: float) -> None:
    print("=" * 78)
    print("SparseFlow ROI Calculator")
    print("=" * 78)

    print("\nWorkload")
    print(f"  GPU Type:            {inp.gpu_type}")
    print(f"  GPU Hourly Cost:     {fmt_money(inp.gpu_hourly_cost)}/hr")
    print(f"  Tokens/Day:          {inp.tokens_per_day:,.0f}")
    print(f"  Tokens/Request:      {inp.tokens_per_request:,.0f}")
    print(f"  Requests/Day:        {res.requests_per_day:,.0f}")
    print(f"  Target Throughput:   {res.target_rps_total:.4f} RPS")

    print("\nPerformance (throughput model)")
    print(f"  Speedup input:       {res.speedup:.2f}×")
    print(f"  Efficiency factor:   {inp.efficiency:.0%} (conservative)")
    print(f"  Dense RPS/GPU:       {res.dense_rps_per_gpu:.4f}")
    print(f"  Sparse RPS/GPU:      {res.sparse_rps_per_gpu:.4f}")

    print("\nGPU Requirements")
    print(f"  Dense GPUs Needed:   {res.dense_gpus_needed}")
    print(f"  Sparse GPUs Needed:  {res.sparse_gpus_needed}")
    print(f"  GPUs Saved:          {res.gpus_saved}")

    print("\nCost")
    print(f"  Dense Monthly Cost:  {fmt_money(res.dense_cost_monthly)}")
    print(f"  Sparse Monthly Cost: {fmt_money(res.sparse_cost_monthly)}")
    print(f"  Monthly Savings:     {fmt_money(res.savings_monthly)}")
    print(f"  Yearly Savings:      {fmt_money(res.savings_yearly)}")

    if monthly_fee > 0:
        net_monthly = res.savings_monthly - monthly_fee
        net_yearly = net_monthly * 12.0
        print("\nPricing / Break-even")
        print(f"  Monthly fee:         {fmt_money(monthly_fee)}")
        print(f"  Net monthly:         {fmt_money(net_monthly)}")
        print(f"  Net yearly:          {fmt_money(net_yearly)}")
        if monthly_fee > res.savings_monthly:
            print("  Break-even:          NOT reached at this workload (fee > savings)")
        else:
            print("  Break-even:          reached (savings >= fee)")

    print("\nNotes")
    print("  - Use measured dense RPS/GPU from your real serving stack (vLLM/TGI/TRT-LLM).")
    print("  - This tool estimates capacity and GPU cost deltas; it does not guarantee production speedups.")
    print("=" * 78)
    print()


def main():
    p = argparse.ArgumentParser(description="SparseFlow ROI calculator (throughput mode)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_thr = sub.add_parser("throughput", help="Use measured dense RPS/GPU (recommended).")
    p_thr.add_argument("--tokens-per-day", type=float, required=True)
    p_thr.add_argument("--tokens-per-request", type=float, required=True)
    p_thr.add_argument("--dense-rps-per-gpu", type=float, required=True, help="Measured dense requests/sec per GPU.")
    p_thr.add_argument("--speedup", type=float, required=True, help="Measured speedup (dense_time / sparse_time).")
    p_thr.add_argument("--gpu-hourly-cost", type=float, required=True)
    p_thr.add_argument("--gpu-type", default="A100 80GB")
    p_thr.add_argument("--efficiency", type=float, default=0.90, help="Conservative factor in [0..1], default 0.90.")
    p_thr.add_argument("--monthly-fee", type=float, default=0.0, help="Optional monthly license/support fee to compute net ROI.")

    args = p.parse_args()

    if args.cmd == "throughput":
        validate_inputs(args.tokens_per_day, args.tokens_per_request, args.dense_rps_per_gpu, args.speedup, args.gpu_hourly_cost)
        inp = ROIInputs(
            gpu_type=args.gpu_type,
            gpu_hourly_cost=args.gpu_hourly_cost,
            tokens_per_day=args.tokens_per_day,
            tokens_per_request=args.tokens_per_request,
            efficiency=clamp(args.efficiency, 0.0, 1.0),
        )
        res = solve_throughput_mode(inp, args.dense_rps_per_gpu, args.speedup)
        print_report(inp, res, monthly_fee=max(0.0, args.monthly_fee))


if __name__ == "__main__":
    main()
