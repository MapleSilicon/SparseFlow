#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace/SparseFlow")

import argparse, torch
from transformers import AutoModelForCausalLM
from sparseflow.nn.policy import SparseFlowPolicy
from tools.llama_surgery_mlp import replace_llama_mlp_perlinear

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--policy", default="policy_efficient.json")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    policy = SparseFlowPolicy()
    policy.load_runtime_policy(args.policy)
    replace_llama_mlp_perlinear(model, policy, verbose=False)

    # Grab ONE replaced Linear (gate_proj) from layer0
    gate = model.model.layers[0].mlp.gate_proj

    H = model.config.hidden_size
    x = torch.randn(args.batch, args.tokens, H, device="cuda", dtype=torch.float16)

    # Run once
    with torch.no_grad():
        y = gate(x)

    print("OK forward ran.")
    # Print key internals if present
    for attr in ["W", "Ws", "weight_sparse", "gate_sparse"]:
        if hasattr(gate, attr):
            obj = getattr(gate, attr)
            print(f"{attr}: type={type(obj)}")

    # Print shape info
    print("x:", tuple(x.shape), "is_contig:", x.is_contiguous(), "stride:", x.stride())
    print("y:", tuple(y.shape), "is_contig:", y.is_contiguous(), "stride:", y.stride())

if __name__ == "__main__":
    main()
