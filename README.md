# SparseFlow

**2× faster GPU inference using NVIDIA 2:4 structured sparsity.**

## Performance (RTX 3090)
```
Matrix Size | Dense (cuBLAS) | SparseFlow | Speedup
------------|----------------|------------|--------
2048×2048   | 60 TFLOPS     | 96 TFLOPS  | 1.60×
4096×4096   | 65 TFLOPS     | 128 TFLOPS | 1.96×
```

**Use SparseFlow for matrices ≥2048. Use dense for smaller sizes.**

## Quick Start
```bash
pip install sparseflow
```
```python
import torch
import sparseflow as sf

# Your model weights (apply 2:4 sparsity pattern first)
W = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
W_sparse = apply_2_4_pattern(W)  # 50% zeros

# Compress once
Wc = sf.compress_2_4(W_sparse)

# Fast inference (2× faster)
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
y = sf.sparse_mm(x, Wc)  # 128 TFLOPS vs 65 TFLOPS dense
```

## When to Use

✅ **Use SparseFlow:** M ≥ 2048 (2× speedup)  
⚠️ **Use dense:** M < 2048 (overhead dominates)

## What is 2:4 Sparsity?

Pattern where 2 out of every 4 consecutive values are zero:
- **50% memory savings**
- **50% fewer operations**
- **2× speedup** at scale
- **Native hardware support** (Ampere+ GPUs)

## Installation
```bash
pip install sparseflow
```

**From source:**
```bash
git clone https://github.com/MapleSilicon/SparseFlow
cd SparseFlow
pip install -e .
```

## Requirements

- NVIDIA GPU with Ampere+ architecture (RTX 30-series, A100, H100)
- PyTorch ≥ 2.0
- CUDA ≥ 11.0
- Python ≥ 3.8

## Benchmark
```bash
python3 benchmarks/bench_dense_vs_sparse.py
```

## API
```python
import sparseflow as sf

# Compress weights with 2:4 pattern
Wc = sf.compress_2_4(W_sparse)

# Fast sparse matrix multiply
y = sf.sparse_mm(x, Wc)
```

## License

MIT

## Citation
```bibtex
@software{sparseflow2025,
  title = {SparseFlow: High-Performance 2:4 Sparse Inference},
  author = {Maple Silicon Inc.},
  year = {2025},
  url = {https://github.com/MapleSilicon/SparseFlow}
}
```

---

**Built by [Maple Silicon Inc.](https://github.com/MapleSilicon)**  
Compiler technology for AI infrastructure.
