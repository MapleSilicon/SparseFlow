#!/usr/bin/env python3
import argparse
import sys

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM


def _import_sparseflow():
    # Keep SparseFlow import optional so this script can still run/print errors nicely.
    try:
        from sparseflow.nn.policy import SparseFlowPolicy
        from sparseflow.nn.surgery import replace_llama_mlp_module
        return SparseFlowPolicy, replace_llama_mlp_module
    except Exception as e:
        raise RuntimeError(f"Could not import SparseFlowPolicy/replace_llama_mlp_module: {e}") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--policy", type=str, default="policy_efficient.json")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--verbose_replace", action="store_true")
    args = ap.parse_args()

    SparseFlowPolicy, replace_llama_mlp_module = _import_sparseflow()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model: {args.model} (dtype={args.dtype})")
    # NOTE: use dtype=... (not torch_dtype) to avoid deprecation warning in some HF versions
    m = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=None,
    )
    m.eval()
    m.to(args.device)

    print(f"Loading SparseFlow policy: {args.policy}")
    policy = SparseFlowPolicy()
    policy.load_runtime_policy(args.policy)

    print("Applying sparse MLP surgery...")
    replaced = replace_llama_mlp_module(m, policy, verbose=args.verbose_replace)
    print(f"Replaced {replaced} MLP modules")

    prompt = "Hello"
    inputs = tok([prompt] * args.batch, return_tensors="pt", padding=True,
                 truncation=True, max_length=args.max_length).to(args.device)

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = m(**inputs)
        if args.device == "cuda":
            torch.cuda.synchronize()

    print(f"\nProfiling Sparse MLP ({args.iters} iterations)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if args.device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(args.iters):
                _ = m(**inputs)
        if args.device == "cuda":
            torch.cuda.synchronize()

    print("\n" + "=" * 70)
    print("SPARSE MLP - TOP 30 KERNELS")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="cuda_time_total" if args.device == "cuda" else "cpu_time_total", row_limit=30))

    # Stack grouped view for suspicious ops
    print("\n" + "=" * 70)
    print("STACKS: aten::to / aten::_to_copy / aten::copy_ / aten::cat / aten::pad")
    print("=" * 70)
    events = ["aten::to", "aten::_to_copy", "aten::copy_", "aten::cat", "aten::pad"]
    ka = prof.key_averages(group_by_stack_n=6)
    for ev in events:
        print(f"\n--- {ev} ---")
        try:
            print(ka.table(sort_by="cpu_time_total", row_limit=25, filter_by_keyword=ev))
        except TypeError:
            # Some torch builds don't support filter_by_keyword in table()
            print(ka.table(sort_by="cpu_time_total", row_limit=60))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
