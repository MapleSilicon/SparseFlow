# SparseFlow

**MLIR-based compiler for structured sparsity optimization**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![MLIR](https://img.shields.io/badge/MLIR-19-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

---

## 🎯 What is SparseFlow?

SparseFlow is an **MLIR compiler infrastructure** that detects and exploits structured sparsity in tensor operations. Our **SPA (Sparsity Propagation Analysis)** pass performs static analysis at compile-time to identify zero computation and generate optimized runtimes.

### Key Achievement

✅ **~4× CPU speedup** on structured sparse matmuls (proven and reproducible)  
✅ **Static analysis** at compile-time (no runtime overhead)  
✅ **2D sparsity tracking** (rows + columns)  
✅ **Production-ready** OpenMP runtime  
✅ **Cross-platform** verified (WSL + GitHub Codespaces)  

---

## 📊 Project Status — SPA v0.6

**Last Updated:** December 2024

### ✅ What Works (Production-Ready)

- **MLIR SPA Pass:** 2D sparsity analysis for `linalg.matmul` (row + column masks)
- **JSON Export:** `spa_sparsity.json` with runtime-ready metadata
- **Python Demos:** Reference implementations for validation
- **C++ OpenMP Runtime:** Production kernel achieving ~4× CPU speedup
- **Cross-Platform:** Verified on WSL (Ubuntu 22.04) and GitHub Codespaces (Ubuntu 24.04)
- **Health Check:** One-command verification (`./quick_check.sh`)
- **Documentation:** Technical overview, pitch deck, benchmarks

### ⚠️ What's Missing (Future Work)

- **GPU Kernels:** No CUDA/ROCm support yet (CPU-only)
- **MLIR Integration:** No automatic lowering to runtime calls
- **Framework Integration:** No PyTorch / ONNX / TensorRT support
- **Dynamic Sparsity:** Only static analysis (no runtime profiling)

### 🎯 Honest Claim

> "SparseFlow SPA v0.6 provides static 2D sparsity analysis for MLIR that detects ~75% removable FLOPs on structured patterns, exports JSON metadata, and drives an OpenMP runtime achieving ~4× CPU speedup on benchmarks from 128×128 to 1024×1024. Verified on WSL and GitHub Codespaces."

---

## 🚀 Quick Start

### Try in GitHub Codespaces (3 minutes)
```bash
# Open this repo in Codespaces, then:

# 1) Health check (builds everything + runs tests)
./quick_check.sh

# 2) See the speedup
cd runtime/build && ./benchmark_sparse
```

**Expected Result:** ~4× speedup on CPU with ~75% sparsity detection

### Local Setup (WSL/Linux)
```bash
# Prerequisites
sudo apt install -y llvm-19-dev mlir-19-tools libmlir-19-dev libomp-dev

# Clone and build
git clone https://github.com/MapleSilicon/SparseFlow.git
cd SparseFlow

# Build compiler passes
cd compiler/build
cmake -DCMAKE_PREFIX_PATH=/usr/lib/llvm-19 .. && make -j4

# Build runtime
cd ../../runtime/build
cmake .. && make -j4

# Run demo
cd ../../
./run_spa_v06_demo.sh
```

---

## 📈 Benchmark Results

### CPU Performance (OpenMP, GitHub Codespaces)

| Matrix Size | Dense Time | Sparse Time | Speedup |
|-------------|------------|-------------|---------|
| 256×256     | 22.3 ms    | 5.2 ms      | **4.3×** |
| 512×512     | 336 ms     | 101 ms      | **3.3×** |
| 768×768     | 745 ms     | 156 ms      | **4.8×** |
| 1024×1024   | 4073 ms    | 945 ms      | **4.3×** |

**Average: 4.2× speedup** (consistent with 75% FLOP reduction)

**Pattern:** 50% row + 50% column sparsity = 75% total sparsity

See [BENCHMARKS.md](BENCHMARKS.md) for detailed methodology and cross-environment results.

---

## 📚 Documentation

- **[3-Minute Demo](QUICK_DEMO.md)** - Prove it works in 3 commands
- **[Technical Overview](docs/SPA_OVERVIEW.md)** - Architecture and examples
- **[Pitch Deck](docs/pitch/SLIDES.md)** - Investor presentation (7 slides)
- **[Benchmarks](BENCHMARKS.md)** - Detailed performance analysis
- **[Health Check](quick_check.sh)** - One-command verification

---

## 🔬 How It Works

### Pipeline Architecture
```
┌─────────────┐
│ MLIR Source │  Standard linalg.matmul
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  SPA Pass   │  Detects: rowmask=[T,F,T,F]
│   (v0.6)    │          colmask=[T,T,F,F]
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ JSON Export │  spa_sparsity.json
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ C++ Runtime │  OpenMP masked matmul
│   (OpenMP)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ ~4× Speedup │  🔥
└─────────────┘
```

### Example

**Input MLIR:**
```mlir
linalg.matmul ins(%A, %B : tensor<512x512xf32>)
```

**After SPA Analysis:**
```mlir
linalg.matmul {
  sparseflow.spa_rowmask = [true, false, true, false, ...],
  sparseflow.spa_colmask = [true, true, false, false, ...]
} ins(%A, %B : tensor<512x512xf32>)
```

**JSON Export:**
```json
{
  "name": "linalg.matmul",
  "row_sparsity_pct": 50,
  "col_sparsity_pct": 50,
  "total_rows": 512,
  "total_cols": 512
}
```

**Runtime:** Uses masks to skip 75% of computation → **3.3× faster**

---

## 🛠️ Repository Structure
```
SparseFlow/
├── compiler/passes/        # MLIR analysis passes
│   ├── spa/               # SPA v0.6 implementation
│   ├── SPAExportPass.cpp  # JSON export
│   └── ...
├── runtime/               # C++ OpenMP runtime
│   ├── masked_matmul.cpp  # Optimized sparse kernel
│   └── benchmark_sparse.cpp
├── docs/
│   ├── SPA_OVERVIEW.md    # Technical deep-dive
│   └── pitch/SLIDES.md    # Investor deck
├── tests/                 # Test cases
├── quick_check.sh         # Health check script
├── run_spa_v06_demo.sh    # Complete demo
└── BENCHMARKS.md          # Performance results
```

---

## 🎓 Technical Details

### What SPA Detects

- **2D Sparsity:** Tracks zero rows AND columns (not just 1D)
- **Static Analysis:** Compile-time detection (no runtime overhead)
- **Structured Patterns:** N:M, block, and custom sparsity
- **Propagation:** Tracks sparsity through arithmetic operations

### Supported Operations (SPA v0.6)

- ✅ `linalg.matmul` (fully supported)
- ✅ `arith.addf`, `arith.subf` (union semantics)
- ✅ `arith.mulf`, `arith.divf` (intersection semantics)
- ✅ `arith.maximumf` (ReLU detection)
- ✅ `linalg.transpose` (swaps rows ↔ cols)
- ✅ `linalg.reduce` (preserves non-reduced dimension)
- ✅ `tensor.expand_shape` (broadcasts pattern)

### Runtime Implementation

- **Language:** C++ with OpenMP
- **Parallelization:** `#pragma omp parallel for`
- **Mask Type:** `std::vector<uint8_t>` (SIMD-friendly)
- **Algorithm:** Extract active block → compute → scatter back

---

## 🚧 Roadmap

### ✅ Phase 1: Static Analysis (Complete)
- 2D sparsity tracking
- JSON export
- CPU runtime
- Cross-platform verification

### 🔨 Phase 2: GPU Acceleration (Next)
- CUDA masked matmul kernel
- 10-50× speedup potential
- cuSPARSE comparison

### 📅 Phase 3: Framework Integration (Future)
- PyTorch plugin
- ONNX Runtime backend
- TensorRT integration

### 🔬 Phase 4: Advanced Features (Research)
- Dynamic sparsity profiling
- Automatic pattern detection
- Multi-dimensional tensors

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- GPU kernel development (CUDA/ROCm)
- MLIR dialect integration
- Framework plugins (PyTorch/ONNX)
- Benchmark suite expansion

---

## 📫 Contact

**Gourav Kumar** - Founder, MapleSilicon  
**GitHub:** [@MapleSilicon](https://github.com/MapleSilicon)  
**Project:** https://github.com/MapleSilicon/SparseFlow

---

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details

---

## 🎉 Acknowledgments

Built with LLVM/MLIR 19. Tested on WSL and GitHub Codespaces.

**Star this repo** ⭐ if you find it useful!

## SPA Runtime (C++ / OpenMP)

SparseFlow includes a minimal C++ runtime that consumes the SPA masks and
accelerates matmuls on CPU:

- Uses **row/column masks** from SPA to skip zero rows/cols
- Implements a **blocked, OpenMP-parallel matmul kernel**
- Achieves **~3–4× speedup** on large matmuls (512–1024) when SPA detects 75% sparsity

### Quick Start

Run the full demo:
```bash
./spa-runner.sh
```

This will run:
* MLIR → SPA → `spa_sparsity.json`
* C++ runtime benchmark with dense vs sparse timings

### Results

On CPU with 50% row + 50% col sparsity (75% FLOP reduction):
- **512×512:** ~3.4× speedup
- **1024×1024:** ~4.9× speedup
- Theoretical maximum: 4.0×

Performance varies with cache effects and OpenMP overhead. Production deployments
should target workloads ≥512×512 for consistent speedup.


## Python CLI (developer preview)

From the repo root:

```bash
cd sparseflow_package
pip install -e .

