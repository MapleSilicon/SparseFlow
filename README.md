# SparseFlow

SparseFlow is an MLIR-based compiler infrastructure for detecting and exploiting **structured N:M sparsity** in linear algebra workloads.

It performs **static compile-time sparsity analysis** and lowers sparse operations through CPU and GPU backends without runtime sparsity checks.

---

## 🔍 What SparseFlow Does

SparseFlow provides an end-to-end compiler pipeline that:

1. Detects structured sparsity (N:M) at compile time
2. Propagates sparsity through the IR
3. Rewrites dense operations into sparse kernel calls
4. Lowers sparse calls to backend-specific implementations (CPU / GPU)

---

## 🧱 Compiler Pipeline (Current)
```
linalg.matmul (dense)
  ↓ sparseflow-spa
Annotated sparse matmul
  ↓ sparseflow-rewrite-matmul
func.call @sparse_matmul_N_M
  ↓ sparseflow-gpu-rewrite
gpu.launch + gpu.func (kernel stub)
```

---

## 📌 Current Release

**Version:** v0.3-alpha  
**Status:** Architectural milestone (GPU lowering path validated)

### Included
- Static N:M sparsity propagation (SPA)
- CPU sparse matmul rewrite
- GPU lowering pass (`gpu.launch` + `gpu.func`)
- Verifier-clean MLIR
- End-to-end pipeline execution

### Not Included (by design)
- GPU kernel implementation
- Bufferization / memory lowering
- Performance claims on GPU

---

## 🚧 Roadmap

- v0.3-beta: GPU kernel ABI + memory mapping
- v0.4: GPU sparse matmul implementation
- v1.0: End-to-end optimized CPU + GPU backend

---

## ⚠️ Disclaimer

SparseFlow v0.3-alpha is a **compiler architecture release**, not a performance release.  
GPU kernels are placeholders for validation only.

---

## 📄 License

MIT
