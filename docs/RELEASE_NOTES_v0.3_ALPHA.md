# SparseFlow v0.3-alpha — Release Notes

## Release Type
**Alpha / Architectural milestone**

## Summary
This release completes SparseFlow's **GPU lowering pipeline**, validating the compiler architecture from sparse analysis to GPU IR generation.

No GPU computation is performed yet.

---

## What's New

### Compiler Passes
- `sparseflow-spa`  
  Static N:M sparsity propagation

- `sparseflow-rewrite-matmul`  
  Rewrites sparse matmul into backend calls

- `sparseflow-gpu-rewrite`  
  Lowers sparse calls to:
  - `gpu.module`
  - `gpu.func`
  - `gpu.launch`

---

## Verified Pipeline
```
linalg.matmul
  → sparseflow-spa
  → func.call @sparse_matmul
  → gpu.launch + gpu.func
```

All passes load correctly via MLIR plugin system.

---

## Known Limitations

- GPU kernel contains only a stub (`gpu.return`)
- No memory lowering or bufferization
- Output tensor is currently a placeholder
- No correctness or performance validation on GPU

---

## Intended Use

- Compiler research
- Backend development
- GPU kernel prototyping
- MLIR-based sparse compiler experimentation

---

## Stability

- IR is verifier-clean
- Pass registration stable
- Plugin loading stable

---

## Next Milestone

**v0.3-beta**
- Kernel ABI definition
- Memory layout decisions
- GPU kernel lowering
