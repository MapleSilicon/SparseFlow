# SparseFlow v0.2.0 - N:M Generalized Sparsity

**Release Date:** December 9, 2025  
**Codename:** "Universal Patterns"

---

## 🎉 Major Features

### N:M Structured Sparsity

SparseFlow now supports **any N:M sparsity pattern**, not just 2:4!

**Supported Patterns:**
- 1:4 (25% density) - Highly sparse networks
- 2:4 (50% density) - NVIDIA Tensor Core compatible
- 2:8 (25% density) - Large sparse models
- 4:16 (25% density) - Memory-constrained inference
- 8:32 (25% density) - Very large models

### Pattern-Aware Compiler

- Automatic N:M pattern detection and propagation
- Pattern-specific runtime kernel selection
- Metadata export with full pattern information

### Runtime Validation

- Pattern validation functions for debugging
- Detailed statistics on pattern conformance
- Zero-threshold configurable validation

---

## 📊 Performance

Same great performance as v0.1, now extended to all patterns:

| Pattern | Density | Expected Speedup |
|---------|---------|------------------|
| 1:4 | 25% | 3.5-4.0× |
| 2:4 | 50% | 1.8-2.2× |
| 2:8 | 25% | 3.5-4.0× |
| 4:16 | 25% | 3.5-4.0× |
| 8:32 | 25% | 3.5-4.0× |

---

## 🔧 What Changed

### Compiler

**SPADomain:**
- Added `NMPattern` struct
- Extended `MatrixSparsity` with pattern info
- New `makeNMPattern()` factory function

**SparsityPropagationPass:**
- Pattern-aware propagation rules
- Preserves N:M structure through operations
- Exports pattern attributes

**SparseMatmulRewritePass:**
- Dynamic function name generation
- Pattern-specific kernel calls
- Attribute propagation

**SPAExportPass:**
- JSON includes N, M, density, pattern name
- Version tracking (v0.2)
- Backwards compatible

### Runtime

**New Functions:**
```c
void sparse_matmul_1_4(...);
void sparse_matmul_2_4(...);  // Same as v0.1
void sparse_matmul_2_8(...);
void sparse_matmul_4_16(...);
void sparse_matmul_8_32(...);
```

**Validation:**
```c
bool validate_nm_pattern(...);
ValidationResult validate_nm_pattern_detailed(...);
```

---

## 📚 Documentation

- Complete N:M API reference
- Migration guide from v0.1
- Usage examples for all patterns
- Troubleshooting guide

---

## ⚡ Quick Start
```mlir
// Annotate with N:M pattern
%sparse = "sparseflow.mark"(%input) {n = 1 : i32, m = 4 : i32}

// Use in matmul
%result = linalg.matmul ins(%sparse, %B) outs(%C)
```
```bash
# Run compiler pipeline
mlir-opt input.mlir \
  --sparseflow-spa \
  --sparseflow-rewrite-matmul \
  -o output.mlir
```

---

## 🔄 Migration from v0.1

✅ **Fully backwards compatible**  
✅ No breaking changes  
✅ v0.1 code works unchanged  

See [MIGRATION_v0.1_to_v0.2.md](docs/v0.2/MIGRATION_v0.1_to_v0.2.md)

---

## 🐛 Bug Fixes

None - v0.1 was already solid!

---

## 🙏 Acknowledgments

Built with:
- MLIR 19.x
- LLVM
- OpenMP
- A lot of caffeine ☕

---

## 📈 What's Next?

**v0.3 (Q2 2026): GPU Acceleration**
- CUDA kernels for N:M patterns
- 5-15× GPU speedup target
- Multi-GPU support

See [ROADMAP.md](ROADMAP.md) for full development plan.

---

## 📫 Contact

- **Email**: maplesilicon1@gmail.com
- **GitHub**: https://github.com/MapleSilicon/SparseFlow
- **Issues**: https://github.com/MapleSilicon/SparseFlow/issues

---

**Thank you for using SparseFlow!** 🚀

*SparseFlow v0.2.0 - Making sparse inference fast and flexible*
