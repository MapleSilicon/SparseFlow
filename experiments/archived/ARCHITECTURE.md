# SparseFlow Compiler Architecture

## Design Goals
- Static sparsity detection
- Zero runtime sparsity overhead
- Backend-agnostic lowering
- Explicit IR stages

---

## Pass Layering

### Analysis
- `sparseflow-spa`

### Rewrite
- `sparseflow-rewrite-matmul`

### Backend Lowering
- `sparseflow-gpu-rewrite`

---

## GPU Design (v0.3-alpha)

- GPU pass does NOT compute
- GPU pass inserts:
  - `gpu.module`
  - `gpu.func`
  - `gpu.launch`
- Kernel body intentionally empty

---

## Why This Matters

This separation allows:
- Independent kernel development
- Clean bufferization
- Multiple GPU strategies without IR rewrites
