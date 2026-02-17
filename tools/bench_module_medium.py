#!/usr/bin/env python3
import sys, time, torch
sys.path.insert(0, "/workspace/SparseFlow")

from transformers import AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy
from tools.llama_surgery_mlp import replace_llama_mlp_module

@torch.no_grad()
def bench(fn, inp, warmup=20, iters=100):
    for _ in range(warmup):
        _ = fn(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(inp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters

def main():
    dtype = torch.float16

    dense = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=dtype
    ).cuda().eval()

    sparse = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=dtype
    ).cuda().eval()

    policy = SparseFlowPolicy()
    policy.load_runtime_policy("policy_efficient.json")
    replaced = replace_llama_mlp_module(sparse, policy, verbose=False)
    print("Replaced modules:", replaced)

    inp = torch.randn(16, 256, 2048, device="cuda", dtype=dtype)

    # smoke
    o = sparse.model.layers[0].mlp(inp)
    print("SPARSE out:", tuple(o.shape), "contig:", o.is_contiguous(), "dtype:", o.dtype)

    td = bench(dense.model.layers[0].mlp, inp)
    ts = bench(sparse.model.layers[0].mlp, inp)

    print("="*60)
    print(f"Dense:  {td:.4f} ms")
    print(f"Sparse: {ts:.4f} ms")
    print(f"Speedup:{td/ts:.3f}×")
    print("="*60)

if __name__ == "__main__":
    main()
