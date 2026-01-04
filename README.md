# SparseFlow

**2× faster GPU inference using NVIDIA 2:4 structured sparsity.**

## Performance (RTX 3090)
```
Matrix Size | Dense (cuBLAS) | SparseFlow | Speedup
------------|----------------|------------|--------
2048×2048   | 60 TFLOPS     | 96 TFLOPS  | 1.60×
4096×4096   | 65 TFLOPS     | 131 TFLOPS | 2.02×
```

## ⚠️ GPU Requirements (CRITICAL)

✅ **Supported:**
- NVIDIA Ampere or newer (SM80+)
- RTX 30-series (3090, 3080, 3070...)
- RTX 40-series (4090, 4080, 4070...)
- A100, A6000, A5000
- H100, H200

❌ **Not Supported:**
- V100 (Volta)
- T4 (Turing)  
- RTX 20-series
- GTX series
- Pascal or older

## Quick Start
```bash
# Install
pip install sparseflow

# Check GPU compatibility FIRST
python3 -c "import sparseflow as sf; print(sf.check_sparse_support())"
```

## Usage
```python
import sparseflow as sf

# Always check compatibility first
if not sf.is_sparse_available():
    print("GPU not supported - use dense operations")
    exit()

# Your sparse inference code here
Wc = sf.compress_2_4(W_sparse)
y = sf.sparse_mm(x, Wc)
```

## When to Use

✅ **Use SparseFlow:** M ≥ 2048 on Ampere+ GPUs  
⚠️ **Use dense:** M < 2048 or older GPUs

## License

MIT

## Version

v2.1.0 - GPU compatibility detection + safe fallbacks
