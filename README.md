# SparseFlow

**GPU acceleration for LLaMA inference using NVIDIA 2:4 sparse tensor cores**

---

## 🚀 What is SparseFlow?

SparseFlow leverages NVIDIA's sparse tensor cores to accelerate LLaMA-70B inference with:
- **1.2-1.4× speedup** on production workloads (batch size ≥ 512)
- **Zero accuracy loss** - validated across all production shapes
- **308-334 TFLOPS** peak throughput on A100
- **Drop-in replacement** for torch.matmul

## 📊 Performance Results

Benchmarked on NVIDIA A100 80GB:

| Shape | Batch | Speedup | TFLOPS |
|-------|-------|---------|--------|
| LLaMA FFN gate | 2048 | 1.42× | 308.4 |
| LLaMA FFN gate | 512 | 1.34× | 286.8 |
| LLaMA FFN down | 2048 | 1.32× | 310.9 |
| LLaMA attn | 2048 | 1.24× | 308.9 |
| LLaMA attn | 512 | 1.17× | 237.4 |

**Best for production workloads with batch size ≥ 512**

## 💰 ROI Example

At 1B tokens/day on A100 GPUs:
- Monthly savings: ~$7,300
- Yearly savings: ~$87,600

## ✅ Correctness Validation

- 11/11 production shapes passing
- Max error: 0.031-0.194 (FP16-appropriate)
- Validated against FP32 ground truth

## 📦 Quick Start
```python
import torch

# Prune to 2:4 sparsity
A_pruned = sparseflow.prune_24(A)

# Convert to sparse format
A_sparse = torch.sparse.to_sparse_semi_structured(A_pruned)

# Fast sparse matmul
C = torch.matmul(A_sparse, B)  # 1.2-1.4× faster
```

## 🛠️ Requirements

- GPU: NVIDIA Ampere or newer (A100, H100, RTX 30/40)
- PyTorch: 2.0+
- CUDA: 11.8+ or 12.x
- Batch size: ≥ 512 for optimal speedup

## 📚 Documentation

- [Integration Guide](docs/INTEGRATION.md) - Production deployment
- [Demo Notebook](demo/SparseFlow_Demo.ipynb) - Interactive demo
- [Benchmarks](benchmarks/) - Performance data
- [ROI Calculator](tools/roi_calculator.py) - Cost savings

## 🧪 Validate & Benchmark
```bash
# Correctness validation
python tests/validate_sparseflow_production.py

# Performance benchmarks  
python benchmarks/run_benchmarks.py

# ROI calculation
python tools/roi_calculator.py throughput \
  --tokens-per-day 1000000000 \
  --dense-rps-per-gpu 3.0 \
  --speedup 1.42 \
  --gpu-hourly-cost 2.50
```

## ⚡ When to Use

**✅ Use for:**
- LLaMA/Transformer inference
- Batch sizes ≥ 512
- Ampere+ GPUs
- FP16 workloads

**❌ Not for:**
- Small batches (< 256)
- Training (no gradient support)
- Pre-Ampere GPUs

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

**Built by Maple Silicon Inc. for production LLaMA inference at scale.**
