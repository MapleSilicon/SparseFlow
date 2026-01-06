# SparseFlow Deployment Guide

## Quick Start

### 1. Check GPU Compatibility
```bash
python3 -c "import sparseflow; print(sparseflow.check_sparse_support())"
```

Requirements:
- NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer)
- CUDA 11.8+
- PyTorch 2.0+

### 2. Analyze Your Deployment
```bash
sparseflow-audit --model llama-7b --qps 1000
```

This shows:
- GPU requirements (dense vs sparse)
- Annual cost savings
- Carbon footprint reduction
- ROI timeline

### 3. Convert Your Model
```bash
sparseflow-convert \
  --input model.pt \
  --output model_sparse.sf \
  --validate
```

Validates 2:4 patterns and reports accuracy impact.

### 4. Benchmark Performance
```bash
sparseflow-benchmark --size 4096x4096 --iterations 100
```

Measures actual speedup on your hardware.

## Production Deployment

### Step 1: Model Conversion
```python
import torch
from torch import nn
import sparseflow as sf

# Load your model
model = torch.load("model.pt")

# Convert Linear layers
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        sparse_layer, diff = sf.SparseLinear.from_dense(
            module,
            method="magnitude",
            return_diff=True
        )
        print(f"{name}: {diff['max_error']:.6f} max error")
        
        # Replace in model
        # parent.layer = sparse_layer
```

### Step 2: Validate Accuracy
```python
# Test on validation set
dense_accuracy = evaluate(dense_model, val_loader)
sparse_accuracy = evaluate(sparse_model, val_loader)

print(f"Accuracy delta: {sparse_accuracy - dense_accuracy:.4f}")
```

Typical accuracy impact: < 0.5% on most tasks

### Step 3: Deploy
```python
# Inference
x = torch.randn(1, 4096, device='cuda', dtype=torch.float16)
y = sparse_model(x)  # 2× faster
```

## Cost Analysis

### Example: LLaMA 7B @ 1000 QPS

**Dense (baseline):**
- GPUs: 16× A100-80GB
- Annual GPU cost: $515K
- Annual power cost: $67K
- Total: $582K/year

**SparseFlow:**
- GPUs: 8× A100-80GB
- Annual GPU cost: $258K
- Annual power cost: $34K
- Total: $292K/year

**Savings:**
- $290K/year (50% reduction)
- 14 tons CO₂/year
- ROI: Immediate

## Troubleshooting

### "CUDA not available"

- Check: `nvidia-smi`
- Install: CUDA Toolkit 11.8+

### "2:4 sparse not supported"

- Requires: Ampere (SM80) or newer
- Check: `torch.cuda.get_device_capability()`

### "Slower than dense"

- Check batch size (need ≥32 for speedup)
- Check GPU utilization
- Try different tile sizes

## Support

- GitHub Issues: https://github.com/MapleSilicon/SparseFlow/issues
- Email: engineering@maplesilicon.com
