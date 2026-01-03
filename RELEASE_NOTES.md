# SparseFlow v2.0 - Production Release

## Performance Summary (RTX 3090)

### Sparse Wins at Scale
```
Size     | Dense    | Sparse (eff) | Real Work | Speedup
---------|----------|--------------|-----------|--------
2048²    | 60 TFLOPS| 96 TFLOPS   | 48 TFLOPS | 1.60×
4096²    | 65 TFLOPS| 128 TFLOPS  | 64 TFLOPS | 1.96×
```

### When to Use Sparse
- ✅ **Use sparse:** M ≥ 2048 (1.6-2× speedup)
- ⚠️ **Use dense:** M < 2048 (overhead dominates)

## Technical Details

### What We Report
- **Effective TFLOPS:** Time to do dense-equivalent work (marketing number)
- **Real Work TFLOPS:** Actual multiply-adds performed (50% of dense)
- **Speedup:** Wall-clock time improvement

### Why Small Sizes Lose
Sparse Tensor Cores have setup overhead:
- Metadata decompression
- Kernel launch latency
- Memory alignment requirements

At small sizes, overhead > savings from 2:4 pattern.

### 2:4 Structured Sparsity
- Pattern: 2 zeros per 4 consecutive values
- Memory: 50% reduction
- Compute: 50% fewer multiply-adds
- Hardware: Native Ampere+ support

## Installation
```bash
git clone https://github.com/MapleSilicon/SparseFlow
cd SparseFlow
pip install -e .
python3 benchmarks/bench_dense_vs_sparse.py
```

## Usage
```python
import sparseflow as sf

# Apply 2:4 pattern to weights
B_sparse = apply_2_4_pattern(B_dense)

# Compress
Bc = sf.compress_2_4(B_sparse)

# Fast inference (M >= 2048)
C = sf.sparse_mm(A, Bc)
```

## Version History

- **v1.0:** 31.62 TFLOPS (Dense WMMA)
- **v1.1:** 32.87 TFLOPS (Dense PTX cp.async)
- **v2.0:** 127.88 TFLOPS effective (2:4 Sparse)

## Requirements

- NVIDIA Ampere+ GPU (RTX 30-series, A100, H100)
- PyTorch with CUTLASS support
- CUDA 11.0+
- Python 3.8+

## Limitations

- Requires pre-applied 2:4 sparsity pattern
- Best for inference (not training)
- Overhead makes it slower for small matrices

## License

MIT

---

**Built by Maple Silicon Inc.**
**Compiler technology for AI infrastructure**
