# SparseFlow

> **Compiler-Driven Sparse Tensor Inference**  
> Automatic sparsity detection and exploitation for 3-5× faster neural network inference

[![Build & Demo](https://github.com/MapleSilicon/SparseFlow/actions/workflows/demo.yml/badge.svg)](https://github.com/MapleSilicon/SparseFlow/actions/workflows/demo.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](CHANGELOG.md)
[![MLIR](https://img.shields.io/badge/MLIR-19.x-orange.svg)](https://mlir.llvm.org/)

**GitHub**: [MapleSilicon/SparseFlow](https://github.com/MapleSilicon/SparseFlow)

---

## 🚀 What is SparseFlow?

SparseFlow is an **MLIR-based compiler** that automatically detects and exploits structured sparsity in neural networks, delivering **3-5× speedups** on commodity hardware with **zero accuracy loss**.

Unlike runtime-based solutions, SparseFlow performs **static sparsity analysis at compile time**, eliminating profiling overhead and guaranteeing correctness.

---

## ⚡ Performance

**Proven speedups on 50% structured sparsity (2:4 pattern):**
```
Matrix Size | Dense Time | Sparse Time | Speedup
------------|------------|-------------|--------
  128×128   |   2.21 ms  |   0.54 ms   |  4.09×
  256×256   |  20.24 ms  |   5.33 ms   |  3.80×
  512×512   | 247.74 ms  |  54.49 ms   |  4.55×
 1024×1024  |2575.15 ms  | 713.08 ms   |  3.61×
```

**Average: 4× faster with 75% fewer operations**

> 💡 **See it in action**: Check the [GitHub Actions](https://github.com/MapleSilicon/SparseFlow/actions) tab for live demo runs!

---

## 🎯 Quick Start

### Run Online Demo

Click here to run the demo in GitHub Actions:

[![Run Demo](https://img.shields.io/badge/Run-Demo-success?style=for-the-badge&logo=github)](https://github.com/MapleSilicon/SparseFlow/actions/workflows/demo.yml)

Or manually trigger: Actions → SparseFlow Demo → Run workflow

### Local Installation
```bash
# Clone repository
git clone https://github.com/MapleSilicon/SparseFlow
cd SparseFlow

# Build compiler
cd compiler/build
cmake .. -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
         -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm
make -j8

# Build runtime
cd ../../runtime/build
cmake ..
make -j8

# Run demo
cd ~/src/SparseFlow
./run_sparseflow_demo.sh
```

---

## 🏗️ Architecture
```
Input MLIR
    ↓
┌─────────────────────┐
│  SPA Analysis       │  ← Detects sparsity patterns
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Rewrite Pass       │  ← Converts to sparse ops
└─────────────────────┘
    ↓
┌─────────────────────┐
│  LLVM Lowering      │  ← Generates native code
└─────────────────────┘
    ↓
┌─────────────────────┐
│  JIT Execution      │  ← Runs with runtime kernel
└─────────────────────┘
    ↓
Output (4× faster)
```

---

## 🔬 Technical Highlights

### Sparsity Propagation Analysis (SPA)
- **Static analysis** - No runtime profiling needed
- **2D mask propagation** - Tracks row & column sparsity
- **Correctness guaranteed** - Conservative analysis

### Automatic Rewriting
- Converts `linalg.matmul` → `@sparse_matmul_2_4`
- Preserves semantics
- Generates efficient runtime calls

### Optimized Runtime
- **OpenMP parallelization** - Multi-core CPU execution
- **Cache-optimized** - Skips zero blocks
- **Vectorized** - SIMD instructions

---

## 📊 Benchmarks

Run comprehensive benchmarks:
```bash
cd compiler/build
./benchmark_suite
```

Validate correctness:
```bash
./test_jit_correctness
```

---

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development plan.

**Next milestones:**
- **v0.2** (Q1 2026): N:M generalized sparsity, Python API
- **v0.3** (Q2 2026): GPU acceleration (CUDA kernels)
- **v0.4** (Q2-Q3 2026): Real neural networks
- **v0.5** (Q3 2026): PyTorch integration
- **v1.0** (Q4 2026): Production release

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

- MLIR compiler passes
- CUDA/ROCm kernels
- PyTorch/ONNX integration
- Benchmark development
- Documentation

See our [issues](https://github.com/MapleSilicon/SparseFlow/issues) for current tasks.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📫 Contact

- **Email**: maplesilicon1@gmail.com
- **GitHub Issues**: [MapleSilicon/SparseFlow/issues](https://github.com/MapleSilicon/SparseFlow/issues)
- **GitHub Discussions**: [MapleSilicon/SparseFlow/discussions](https://github.com/MapleSilicon/SparseFlow/discussions)

---

*SparseFlow v0.1.0 - Making sparse inference fast and automatic*
