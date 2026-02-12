import argparse, sys, torch
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy
from sparseflow.nn.sparseflow_linear import SparseFlowLinear, make_sparseflow_linear

def replace_llama_mlp(model, policy):
    """Replace MLP layers with SparseFlow"""
    replaced = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(mlp, proj_name, None)
                if lin and isinstance(lin, torch.nn.Linear):
                    setattr(mlp, proj_name, make_sparseflow_linear(lin, policy=policy, op_name=proj_name))
                    replaced += 1
    return replaced

def count_modules(model):
    n_lin = sum(isinstance(m, nn.Linear) for m in model.modules())
    n_sfl = sum(isinstance(m, SparseFlowLinear) for m in model.modules())
    return n_lin, n_sfl

@torch.no_grad()
def run_prefill(model, inp):
    return model(**inp)

def cuda_ms(fn, iters=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--policy_json", default="policy_efficient.json")
    ap.add_argument("--mode", default="max-autotune")
    args = ap.parse_args()

    device="cuda"
    dtype=torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("="*70)
    print("4-WAY BENCHMARK: The Truth Test")
    print("="*70)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print(f"Batch={args.batch} Seq={args.seq} M={args.batch*args.seq}")
    print("="*70)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    inp = tok(["Hello, write a short story about Canada."] * args.batch,
              return_tensors="pt", padding=True, truncation=True, max_length=args.seq)
    inp = {k: v.to(device) for k, v in inp.items()}

    # Load models
    print("\nLoading models...")
    dense = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    sparse = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    policy = SparseFlowPolicy()
    policy.load_runtime_policy(args.policy_json)
    n_replaced = replace_llama_mlp(sparse, policy)

    print(f"[Dense ] nn.Linear={count_modules(dense)[0]}, SparseFlowLinear={count_modules(dense)[1]}")
    print(f"[Sparse] nn.Linear={count_modules(sparse)[0]}, SparseFlowLinear={count_modules(sparse)[1]} (replaced {n_replaced})")

    # Eager
    print("\nBenchmarking eager...")
    dense_e = cuda_ms(lambda: run_prefill(dense, inp), iters=args.iters, warmup=args.warmup)
    sparse_e = cuda_ms(lambda: run_prefill(sparse, inp), iters=args.iters, warmup=args.warmup)

    # Compiled
    print(f"Compiling (mode={args.mode})...")
    dense_c = torch.compile(dense, mode=args.mode)
    sparse_c = torch.compile(sparse, mode=args.mode)

    print("Benchmarking compiled...")
    dense_ce = cuda_ms(lambda: run_prefill(dense_c, inp), iters=args.iters, warmup=args.warmup)
    sparse_ce = cuda_ms(lambda: run_prefill(sparse_c, inp), iters=args.iters, warmup=args.warmup)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Dense eager      : {dense_e:8.3f} ms")
    print(f"Sparse eager     : {sparse_e:8.3f} ms  ({dense_e/sparse_e:5.2f}× vs dense eager)")
    print(f"Dense compiled   : {dense_ce:8.3f} ms  ({dense_e/dense_ce:5.2f}× vs dense eager)")
    print(f"Sparse compiled  : {sparse_ce:8.3f} ms  ({dense_ce/sparse_ce:5.2f}× vs dense compiled) ← KEY METRIC")
    print("="*70)
    
    if dense_ce / sparse_ce > 1.05:
        print("✅ WIN: SparseFlow adds value on top of torch.compile")
    elif dense_ce / sparse_ce > 0.95:
        print("≈ NEUTRAL: SparseFlow ≈ torch.compile baseline")
    else:
        print("❌ LOSS: SparseFlow slower than torch.compile baseline")
    print("="*70)

if __name__ == "__main__":
    main()
