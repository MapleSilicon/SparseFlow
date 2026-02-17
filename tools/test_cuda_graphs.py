#!/usr/bin/env python3
import sys, time, torch
sys.path.insert(0, "/workspace/SparseFlow")

from transformers import AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy
from tools.llama_surgery import replace_llama_mlp

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class GraphRunner:
    def __init__(self, fn, static_inp, warmup=30):
        self.fn = fn
        self.static_inp = static_inp
        self.static_out = None

        # Use dedicated graph pool
        self.pool = torch.cuda.graphs.graph_pool_handle()
        self.g = torch.cuda.CUDAGraph()

        # Warmup outside capture
        for _ in range(warmup):
            _ = self.fn(self.static_inp)
        torch.cuda.synchronize()

        # Capture
        torch.cuda.synchronize()
        with torch.cuda.graph(self.g, pool=self.pool):
            self.static_out = self.fn(self.static_inp)
        torch.cuda.synchronize()

    def replay(self):
        self.g.replay()
        return self.static_out

@torch.no_grad()
def bench(fn, inp, iters=200, warmup=50):
    for _ in range(warmup):
        _ = fn(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(inp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters

@torch.no_grad()
def bench_graph(fn, inp, iters=400, warmup=30):
    gr = GraphRunner(fn, inp, warmup=warmup)
    for _ in range(50):
        gr.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        gr.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters

def load_model(dtype):
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).cuda().eval()
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).cuda().eval()

print("="*80)
print("CUDA GRAPH TEST - MEDIUM (T=256, B=16)")
print("="*80)

dtype = torch.float16

dense = load_model(dtype)
sparse = load_model(dtype)

policy = SparseFlowPolicy()
policy.load_runtime_policy("policy_efficient.json")
replace_llama_mlp(sparse, policy, verbose=False)

dense_mlp = dense.model.layers[0].mlp
sparse_mlp = sparse.model.layers[0].mlp

inp = torch.randn(16, 256, dense.config.hidden_size, device="cuda", dtype=dtype)

print("Benchmarking baseline...")
td_base = bench(dense_mlp, inp)
ts_base = bench(sparse_mlp, inp)

print("Capturing CUDA graphs...")
td_graph = bench_graph(dense_mlp, inp)
ts_graph = bench_graph(sparse_mlp, inp)

print("")
print(f"Dense baseline:   {td_base:.4f} ms")
print(f"Dense graph:      {td_graph:.4f} ms   ({td_base/td_graph:.3f}× faster)")
print(f"Sparse baseline:  {ts_base:.4f} ms")
print(f"Sparse graph:     {ts_graph:.4f} ms   ({ts_base/ts_graph:.3f}× faster)")
print("")
print(f"Baseline speedup: {td_base/ts_base:.3f}×")
print(f"Graph speedup:    {td_graph/ts_graph:.3f}× 🎯")
print("="*80)
