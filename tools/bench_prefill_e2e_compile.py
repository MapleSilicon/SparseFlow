import argparse, time, os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.nn.sparseflow_linear import SparseFlowLinear

# These are in your repo (you already validated they exist)
from tools.llama_surgery import replace_llama_mlp
from tools.llama_surgery_full import replace_llama_full

def bench(model, inp, iters, warmup):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(**inp)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(**inp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters

def maybe_compile(model, do_compile: bool, mode: str):
    if not do_compile:
        return model
    # fullgraph=False is safer for weird ops
    return torch.compile(model, mode=mode, fullgraph=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--mode", type=str, default="max-autotune")
    ap.add_argument("--policy", type=str, default="policy_efficient.json")
    ap.add_argument("--full", action="store_true", help="replace attention too (replace_llama_full)")
    args = ap.parse_args()

    print("="*70, flush=True)
    print("SparseFlow bench_prefill_e2e_compile", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    print(f"model={args.model} batch={args.batch} seqlen={args.max_len} iters={args.iters} warmup={args.warmup}", flush=True)
    print(f"compile={args.compile} mode={args.mode} full_replace={args.full}", flush=True)
    print("="*70, flush=True)

    device = "cuda"
    dtype = torch.float16

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Build fixed-size random-ish input ids (no generation, just prefill forward)
    # Use tokenizer to get vocab size safely
    vocab = int(getattr(tok, "vocab_size", 32000))
    input_ids = torch.randint(0, vocab, (args.batch, args.max_len), device=device)
    attn = torch.ones((args.batch, args.max_len), device=device, dtype=torch.long)
    inp = {"input_ids": input_ids, "attention_mask": attn}

    # ---------------- Dense ----------------
    print("\n[1] Loading dense...", flush=True)
    dense = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    print("[1] Benchmark dense eager...", flush=True)
    t_dense = bench(dense, inp, args.iters, args.warmup)
    print(f"    dense eager:    {t_dense:.3f} ms", flush=True)

    print("[1] Benchmark dense compiled...", flush=True)
    dense_c = maybe_compile(dense, args.compile, args.mode)
    t_dense_c = bench(dense_c, inp, args.iters, args.warmup)
    print(f"    dense compiled: {t_dense_c:.3f} ms", flush=True)

    # ---------------- SparseFlow ----------------
    print("\n[2] Loading sparse base...", flush=True)
    sparse = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    policy = SparseFlowPolicy()
    if os.path.exists(args.policy):
        try:
            policy.load_runtime_policy(args.policy)
            print(f"[2] Loaded policy: {args.policy}", flush=True)
        except Exception as e:
            print(f"[2] Policy load failed ({args.policy}): {e}", flush=True)

    print("[2] Applying surgery...", flush=True)
    if args.full:
        replaced = replace_llama_full(sparse, policy, verbose=True)
    else:
        replaced = replace_llama_mlp(sparse, policy, verbose=True)

    n_lin = sum(isinstance(m, nn.Linear) for m in sparse.modules())
    n_sfl = sum(isinstance(m, SparseFlowLinear) for m in sparse.modules())
    print(f"[2] Coverage: nn.Linear={n_lin} SparseFlowLinear={n_sfl} (replaced={replaced})", flush=True)

    print("[2] Benchmark sparse eager...", flush=True)
    t_sparse = bench(sparse, inp, args.iters, args.warmup)
    print(f"    sparse eager:   {t_sparse:.3f} ms", flush=True)

    print("[2] Benchmark sparse compiled...", flush=True)
    sparse_c = maybe_compile(sparse, args.compile, args.mode)
    t_sparse_c = bench(sparse_c, inp, args.iters, args.warmup)
    print(f"    sparse compiled:{t_sparse_c:.3f} ms", flush=True)

    print("\n" + "="*70, flush=True)
    print("RESULTS (lower is better)", flush=True)
    print(f"dense eager        : {t_dense:.3f} ms", flush=True)
    print(f"dense compiled     : {t_dense_c:.3f} ms  ({t_dense/t_dense_c:.2f}× vs dense eager)", flush=True)
    print(f"sparse eager       : {t_sparse:.3f} ms  ({t_dense/t_sparse:.2f}× vs dense eager)", flush=True)
    print(f"sparse compiled    : {t_sparse_c:.3f} ms  ({t_dense/t_sparse_c:.2f}× vs dense eager)", flush=True)
    print(f"key metric (sparse compiled / dense compiled): {t_sparse_c/t_dense_c:.3f}×", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
