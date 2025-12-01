# SparseFlow

Custom MLIR-based compiler for **N:M structured sparsity** acceleration.

- 🎯 Target pattern: **2:4 sparsity (50% dense)**
- ⚙️ Stack: **MLIR / LLVM 19** + custom passes
- 📦 Output: **hardware_config.json** with matmul metadata (M, N, K, sparsity, MAC counts)
- 🧪 Runtime: C++ test binary that programs a simulated "MapleSilicon" sparse accelerator

---

## Status

- ✅ MLIR pass plugin (`SparseFlowPasses.so`) builds with LLVM/MLIR 19
- ✅ Custom passes:
  - `sparseflow-annotate-nm`
  - `sparseflow-export-metadata`
  - `sparseflow-flop-counter`
- ✅ `sparseflow-export-metadata` emits `hardware_config.json`
- ✅ Runtime reads JSON and configures the accelerator model
- ✅ End-to-end pipeline: **MLIR → JSON → Runtime**

---

## Requirements

- Ubuntu 22.04 (or similar)
- LLVM / MLIR 19 (packages provide `mlir-opt-19`, `llvm-config-19`, etc.)
- CMake, Make or Ninja
- C++17 compiler (e.g., GCC 11)

On typical Ubuntu with LLVM 19:

```bash
sudo apt install llvm-19 llvm-19-dev mlir-19-tools clang-19 \
                 libmlir-19-dev cmake ninja-build g++ python3

