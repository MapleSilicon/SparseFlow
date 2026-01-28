#!/usr/bin/env python3
"""
End-to-end prefill benchmark (dense vs pruned-dense vs SparseFlow).
"""
import argparse, time, warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparseflow.nn import make_sparseflow_linear, prune_24_dense_weight, SparseFlowPolicy

torch.backends.cuda.matmul.allow_tf32 = False
warnings.filterwarnings('ignore', category=UserWarning, message='.*to_sparse_semi_structured.*')

def replace_llama_mlp(model, policy: SparseFlowPolicy):
    """Replace MLP linears with SparseFlowLinear"""
    replaced = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, proj_name, None)
            if lin is None or not isinstance(lin, torch.nn.Linear):
                continue
            sfl = make_sparseflow_linear(lin, policy=policy, op_name=proj_name)
            setattr(mlp, proj_name, sfl)
            replaced += 1
    print(f"  Replaced {replaced} Linear layers with SparseFlowLinear")
    return replaced

@torch.no_grad()
def bench(model, tok, prompt, batch, max_len, iters, warmup):
    inputs = tok([prompt]*batch, return_tensors="pt", padding=True, max_length=max_len, truncation=True).to("cuda")

    # warmup
    for _ in range(warmup):
        _ = model(**inputs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(**inputs)
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) * 1000 / iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--min-M-gateup", type=int, default=256)
    ap.add_argument("--min-M-down", type=int, default=512)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading dense model...")
    model_dense = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    print("Loading pruned-dense model...")
    model_pruned = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    for n, m in model_pruned.named_modules():
        if hasattr(m, "weight") and isinstance(m, torch.nn.Linear):
            with torch.no_grad():
                m.weight.copy_(prune_24_dense_weight(m.weight))

    print("Loading SparseFlow model...")
    model_sparse = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    policy = SparseFlowPolicy(min_M_gate_up=args.min_M_gateup, min_M_down=args.min_M_down)
    replace_llama_mlp(model_sparse, policy)

    prompt = "Explain sparse tensor cores in one paragraph."
    
    print(f"\nBenchmarking prefill (batch={args.batch}, seq_len={args.max_len})...")
    td = bench(model_dense, tok, prompt, args.batch, args.max_len, args.iters, args.warmup)
    print(f"  Dense: {td:.2f} ms")
    
    tp = bench(model_pruned, tok, prompt, args.batch, args.max_len, args.iters, args.warmup)
    print(f"  Pruned-dense: {tp:.2f} ms")
    
    ts = bench(model_sparse, tok, prompt, args.batch, args.max_len, args.iters, args.warmup)
    print(f"  SparseFlow: {ts:.2f} ms")

    print("\n=== Results ===")
    print(f"Dense:        {td:.3f} ms")
    print(f"Pruned-dense: {tp:.3f} ms")
    print(f"SparseFlow:   {ts:.3f} ms")
    print(f"\nSpeedup (sparse vs pruned-dense): {tp/ts:.2f}×")
    print(f"Speedup (sparse vs dense):        {td/ts:.2f}×")

if __name__ == "__main__":
    main()
