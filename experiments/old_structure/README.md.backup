# SparseFlow

**MLIR-based compiler for N:M structured sparsity acceleration**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![MLIR](https://img.shields.io/badge/MLIR-19-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

## 🚀 New: Sparsity Propagation Analysis (SPA) v0.5

SparseFlow now includes **SPA** - a static analysis pass that tracks and propagates sparsity through computation graphs:
```bash
# Try it now - one command!
./scripts/run_spa_demo.sh
```

**What it does:**
- 🎯 Converts N:M sparsity to row-level masks
- 📊 Propagates sparsity through arithmetic operations
- ⚡ Enables compile-time optimizations
- 🔗 Integrates with existing MLIR passes

**Documentation:** [docs/spa/](docs/spa/) | **Status:** [SPA_STATUS.md](SPA_STATUS.md)

---

## Quick Start
```bash
git clone https://github.com/MapleSilicon/SparseFlow.git
cd SparseFlow
./build_all.sh
```

**Result:** Proven 2.0x speedup with 2:4 sparsity (50% compute reduction)

## What is SparseFlow?

SparseFlow is a production-ready compiler that optimizes AI inference through structured sparsity:

- **Input:** Standard MLIR from PyTorch/ONNX/TensorFlow
- **Transform:** Apply 2:4 structured sparsity patterns + static analysis
- **Output:** Hardware-ready JSON metadata + optimized IR
- **Result:** 2x theoretical speedup, 50% MACs eliminated

## Key Features

### 1. Sparsity Propagation Analysis (SPA)

Static analysis that tracks which matrix rows/columns are provably zero:
```mlir
// Before SPA
linalg.matmul ins(%A, %B : tensor<4x4xf32>)

// After SPA (with 2:4 sparsity)
linalg.matmul {sparseflow.spa_rowmask = [true, true, false, false]}
  ins(%A, %B : tensor<4x4xf32>)
```

Now the compiler knows rows 2 and 3 are zero! See [SPA Documentation](docs/spa/).

### 2. N:M Sparsity Patterns

Proven 2:4 structured sparsity with consistent performance:

| Matrix Size | Total MACs | Executed MACs | Speedup | Savings |
|-------------|------------|---------------|---------|---------|
| 32×32       | 32,768     | 16,384        | 2.0x    | 50%     |
| 128×128     | 2,097,152  | 1,048,576     | 2.0x    | 50%     |
| 512×512     | 134,217,728| 67,108,864    | 2.0x    | 50%     |

See [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md) for detailed benchmarks.

### 3. End-to-End Pipeline
```
MLIR Input → Annotate N:M → SPA Analysis → Optimize → JSON Export → Runtime
```

## Architecture
```
┌────────────────────────────────────────┐
│         SparseFlow Compiler            │
│                                        │
│  ┌──────────────┐  ┌──────────────┐  │
│  │ AnnotateNm   │─▶│  SPA Pass    │  │
│  │ (N:M inject) │  │  (Analysis)  │  │
│  └──────────────┘  └──────┬───────┘  │
│                            │          │
│  ┌──────────────┐  ┌──────▼───────┐  │
│  │ FlopCounter  │  │ Optimization │  │
│  │              │  │ Passes       │  │
│  └──────────────┘  └──────┬───────┘  │
│                            │          │
│                    ┌───────▼───────┐  │
│                    │ JSON Export   │  │
│                    └───────────────┘  │
└────────────────────────────────────────┘
```

### Compiler Passes

1. **`sparseflow-annotate-nm`** - Inject N:M sparsity patterns
2. **`sparseflow-spa`** - Analyze and propagate sparsity ⭐ NEW
3. **`sparseflow-flop-counter`** - Compute MAC reduction metrics
4. **`sparseflow-export-metadata`** - Generate hardware config JSON

## Repository Structure
```
SparseFlow/
├── compiler/
│   ├── passes/
│   │   ├── AnnotateNmPass.cpp
│   │   ├── FlopCounterPass.cpp
│   │   ├── ExportMetadataPass.cpp
│   │   └── spa/                    ⭐ NEW: SPA implementation
│   │       ├── SPADomain.h
│   │       ├── SPADomain.cpp
│   │       └── SparsityPropagationPass.cpp
│   └── test/
├── docs/                            ⭐ NEW: Complete documentation
│   └── spa/
│       ├── README.md
│       ├── architecture.md
│       ├── user_guide.md
│       ├── api_reference.md
│       └── examples/
├── runtime/
├── benchmarks/
├── scripts/
│   └── run_spa_demo.sh             ⭐ NEW: One-command demo
├── tests/
│   └── spa_nm_demo.mlir            ⭐ NEW: SPA test cases
├── build_all.sh
├── run_benchmarks.sh
├── QUICK_DEMO.md
├── SPA_STATUS.md                   ⭐ NEW: Development status
└── README.md
```

## Getting Started

### Prerequisites

- LLVM/MLIR 19
- CMake 3.20+
- C++17 compiler
- Python 3.8+ (for benchmarks)

### Build
```bash
cd SparseFlow/compiler
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/lib/llvm-19 ..
make -j4
```

### Try SPA
```bash
# Run the demo
./scripts/run_spa_demo.sh

# Or manually
cd compiler/build
mlir-opt-19 --load-pass-plugin=./passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(func.func(sparseflow-annotate-nm),sparseflow-spa)' \
  ../../tests/spa_nm_demo.mlir
```

### Run Benchmarks
```bash
# Full benchmark suite
./run_benchmarks.sh

# Generate performance graphs
python3 generate_graphs.py benchmarks/results/TIMESTAMP/benchmark_results.csv
```

## Documentation

- **[Quick Demo](QUICK_DEMO.md)** - Get started in 5 minutes
- **[SPA Documentation](docs/spa/)** - Complete guide to Sparsity Propagation Analysis
- **[SPA Status](SPA_STATUS.md)** - Development roadmap and features
- **[Performance Results](PERFORMANCE_RESULTS.md)** - Detailed benchmarks
- **[Benchmarks](BENCHMARKS.md)** - Methodology and analysis

## Technical Details

**Sparsity Pattern:** 2:4 structured (2 non-zero values per 4 elements)  
**Analysis:** Static row-level sparsity propagation (v0.5)  
**Target Hardware:** FPGA, ASIC, specialized accelerators  
**Compiler Infrastructure:** MLIR 19, LLVM toolchain  
**Metadata Format:** JSON (hardware-agnostic)

## Status

### Compiler (Production Ready)
- ✅ N:M annotation pass
- ✅ Sparsity propagation analysis (SPA v0.5)
- ✅ FLOP counter
- ✅ Metadata export
- ✅ Pass pipeline validated

### Documentation
- ✅ User guide
- ✅ Architecture documentation
- ✅ API reference
- ✅ Examples and tutorials

### Future Work
- 🔨 SPA v0.6: 2D sparsity (rows + columns)
- 🔨 Runtime integration
- 🔨 FPGA backend
- 🔨 PyTorch integration (Q1 2026)

## Contributing

Contributions welcome! See [docs/](docs/) for technical details.

## Contact

**Gourav Kumar** - Founder, MapleSilicon  
GitHub: [@MapleSilicon](https://github.com/MapleSilicon)

## License

Apache 2.0

---

**Latest:** SPA v0.5 with N:M integration and comprehensive documentation (December 2024)
