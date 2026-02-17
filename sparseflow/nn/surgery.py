import torch
import torch.nn as nn

from sparseflow.nn.sparseflow_mlp import SparseFlowMLP


def _get_mlp_projections(mlp: nn.Module):
    """
    Support common Llama-family layouts:
      - Transformers LlamaMLP: gate_proj, up_proj, down_proj
      - Some variants: w1, w3, w2 (gate, up, down)
    Returns (gate, up, down) Linear modules.
    """
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        return mlp.gate_proj, mlp.up_proj, mlp.down_proj
    if hasattr(mlp, "w1") and hasattr(mlp, "w3") and hasattr(mlp, "w2"):
        # common alt naming: w1=gate, w3=up, w2=down
        return mlp.w1, mlp.w3, mlp.w2
    raise RuntimeError(f"Unsupported MLP layout: {type(mlp)} has attrs {dir(mlp)}")


def replace_llama_mlp_module(model: nn.Module, policy, verbose: bool = False):
    """
    Replace each decoder-layer MLP with SparseFlowMLP using the layer's existing weights.
    Designed for HF Llama-like models (TinyLlama included).
    """
    # Try to find HF-style layers list
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise RuntimeError("Could not locate transformer layers (expected model.model.layers or model.layers).")

    # Sizes from config when possible
    cfg = getattr(model, "config", None)
    cfg_hidden = getattr(cfg, "hidden_size", None)
    cfg_inter = getattr(cfg, "intermediate_size", None)

    replaced = 0

    for i, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue

        mlp = layer.mlp
        gate, up, down = _get_mlp_projections(mlp)

        hidden_size = cfg_hidden
        intermediate_size = cfg_inter

        # If config missing, infer from shapes
        if hidden_size is None:
            hidden_size = gate.in_features
        if intermediate_size is None:
            intermediate_size = gate.out_features

        # Pull weights (+ bias if present)
        # Pull weights (+ bias) and CLONE them into normal tensors.
        # HF can construct these under inference_mode, which breaks downstream ops.
        with torch.inference_mode(False):
            gate_w = gate.weight.detach().clone().contiguous()
            up_w   = up.weight.detach().clone().contiguous()
            down_w = down.weight.detach().clone().contiguous()

            gate_b = getattr(gate, "bias", None)
            up_b   = getattr(up, "bias", None)
            down_b = getattr(down, "bias", None)

            if gate_b is not None: gate_b = gate_b.detach().clone().contiguous()
            if up_b   is not None: up_b   = up_b.detach().clone().contiguous()
            if down_b is not None: down_b = down_b.detach().clone().contiguous()

        # Device/dtype: follow existing weights
        device = str(gate_w.device)
        dtype  = gate_w.dtype

        new_mlp = SparseFlowMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gate_weight=gate_w,
            up_weight=up_w,
            down_weight=down_w,
            gate_bias=gate_b,
            up_bias=up_b,
            down_bias=down_b,
            policy=policy,
            device=device,
            dtype=dtype,
        )

        layer.mlp = new_mlp
        replaced += 1

        if verbose:
            print(f"  [✓] Layer {i:02d} mlp → SparseFlowMLP")

    return replaced
