# Week 4 – SPA v0.6 Export + FLOP Counter Status

**Build**

- mlir-opt-19 using MLIR/LLVM 19
- Pass plugin: compiler/build/passes/SparseFlowPasses.so
- Built via: `cd compiler && rm -rf build && mkdir build && cd build && cmake .. -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm && make -j8`

**Key Passes**

- `sparseflow-spa` (module-level sparsity propagation)
- `sparseflow-spa-export` (JSON export to `spa_sparsity.json`)
- `sparseflow-flop-counter` (func-level FLOP counter for linalg.matmul)

**Demo Pipelines**

1. SPA + Export (from repo root):

   ```bash
   ./spa_v06_export.sh

