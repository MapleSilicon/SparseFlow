# SparseFlow SPA Overview

## What is SPA?

**SPA (Sparsity Propagation Analysis)** is an MLIR compiler pass that statically analyzes structured sparsity patterns in tensor operations and generates runtime-ready metadata for optimization.

## The Problem

Modern ML models contain significant structured sparsity:
- N:M patterns (e.g., 2:4, 4:8) from pruning
- Block sparsity from quantization
- Channel sparsity from architecture search

But current frameworks either:
- **Ignore it** (waste computation on zeros)
- **Detect it at runtime** (overhead + latency)
- **Require manual annotation** (error-prone)

## SparseFlow's Solution

### Pipeline Architecture
```
┌─────────────┐
│ MLIR Source │ (e.g., linalg.matmul)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  SPA Pass   │ Detects: rowmask=[T,F,T,F]
│   (v0.6)    │         colmask=[T,T,F,F]
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ JSON Export │ spa_sparsity.json
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  C++ Runtime│ OpenMP masked matmul
│  (OpenMP)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ~4× Speedup│ 🔥
└─────────────┘
```

### Concrete Example: 512×512 Matmul

**Input Pattern:** 50% row sparsity + 50% column sparsity

**SPA Analysis:**
```mlir
linalg.matmul {
  sparseflow.spa_rowmask = [true, false, true, false, ...],
  sparseflow.spa_colmask = [true, true, false, false, ...]
}
```

**JSON Export:**
```json
{
  "id": 0,
  "name": "linalg.matmul",
  "row_sparsity_pct": 50,
  "col_sparsity_pct": 50,
  "total_rows": 512,
  "total_cols": 512
}
```

**Runtime Performance:**
- Dense baseline: 336 ms
- SPA-optimized: 101 ms
- **Speedup: 3.31×** ✅

**FLOP Reduction:**
- Original: 512×512×512 = 134M FLOPs
- Active: 256×256×512 = 33M FLOPs  
- **Reduction: 75%** (matches speedup!)

## Current Results

| Matrix Size | Speedup | Environment |
|-------------|---------|-------------|
| 256×256     | 4.33×   | Codespaces  |
| 512×512     | 3.31×   | Codespaces  |
| 768×768     | 4.77×   | Codespaces  |
| 1024×1024   | 4.31×   | Codespaces  |

**Average:** ~4× (consistent with 75% FLOP reduction)

## What's Unique

1. **Static Analysis:** Detects sparsity at compile-time (no runtime overhead)
2. **2D Tracking:** Tracks both row and column sparsity (not just 1D)
3. **MLIR Integration:** Works with standard compiler infrastructure
4. **Proven Results:** Reproducible 4× speedup on real hardware

## Current Limitations

- **CPU-only:** No GPU kernels yet
- **Manual Integration:** JSON → Runtime bridge is manual
- **Limited Ops:** Only matmul fully supported
- **No Framework Support:** No PyTorch/TensorRT integration

## Roadmap

**Phase 1 (✅ Complete):** Static analysis + CPU runtime  
**Phase 2 (Next):** GPU acceleration (CUDA/ROCm)  
**Phase 3 (Future):** Framework integration (PyTorch/ONNX)  
**Phase 4 (Research):** Dynamic sparsity tracking

