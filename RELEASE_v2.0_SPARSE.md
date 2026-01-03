# 🏆 SparseFlow v2.0 - Sparse Tensor Core Release

## BREAKTHROUGH PERFORMANCE

### 126.81 TFLOPS on RTX 3090! 🎉

Using 2:4 structured sparsity with native Tensor Core support.

## Results
```
Size      | Dense (cuBLAS) | Sparse (2:4) | Speedup
----------|----------------|--------------|--------
512×512   |  17.63 TFLOPS  |   2.95 TFLOPS| 0.17×
1024×1024 |  47.89 TFLOPS  |  20.36 TFLOPS| 0.43×
2048×2048 |  59.93 TFLOPS  |  92.40 TFLOPS| 1.54×
4096×4096 |  65.53 TFLOPS  | 126.81 TFLOPS| 1.94×
```

## What is 2:4 Sparsity?

In every group of 4 consecutive values, exactly 2 are zero.
- **50% fewer values stored**
- **2× theoretical speedup**
- **Native hardware support on Ampere+**

## Usage
```python
import torch

# Apply 2:4 sparsity pattern
A_sparse = create_2_4_pattern(A_dense)

# Compress
A_compressed = torch._cslt_compress(A_sparse)

# Fast sparse matmul
C = torch._cslt_sparse_mm(A_compressed, B)
```

## Version History

- **v1.0:** 31.62 TFLOPS (Dense WMMA)
- **v1.1:** 32.87 TFLOPS (Dense PTX cp.async)
- **v2.0:** 126.81 TFLOPS (2:4 Sparse) ✨

## Requirements

- RTX 3090 or newer (Ampere+)
- PyTorch with CUTLASS sparse support
- CUDA 11.0+

## Limitations

- Requires 2:4 sparse pattern in weights
- Best for inference (pre-pruned models)
- Quality depends on sparsity-aware training

## Future Work

- Sparse training integration
- Auto-pruning pipelines
- Multi-GPU scaling
- FP8 sparse (H100)

---

**126 TFLOPS. On consumer hardware. 🚀**
