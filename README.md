# 🌲 SparseFlow v0.2.0  
### Generalized N:M Sparse Compiler for AI Inference (MLIR + CPU Runtime)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![MLIR](https://img.shields.io/badge/MLIR-LLVM19-blue)
![Sparsity](https://img.shields.io/badge/Sparsity-N:M-ff69b4)

SparseFlow is an MLIR-based compiler that performs static sparsity analysis and rewrites dense matmuls into highly optimized **generalized N:M sparse kernels**.

Unlike existing solutions restricted to **2:4 only**, SparseFlow supports:

### **1:4, 2:4, 2:8, 4:16, 8:32 — all fully automatic.**

---

## 🚀 Key Features
- 🔍 **Static sparsity detection (SPA Pass)**
- 🔁 **Automatic rewrite into sparse runtime kernels**
- ⚡ **Hand-optimized CPU kernels using OpenMP**
- 🧩 **Generalized N:M pattern support**
- 🧱 **MLIR-native design, clean extensible architecture**

---

## 📊 Performance (REAL Measurements)

SparseFlow achieves **9×–20×** stable speedup on CPU across realistic matrix sizes.

| Matrix | Min Speedup | Max Stable Speedup | Peak (non-stable) |
|--------|-------------|---------------------|-------------------|
| 256×256 | 3× | 8× | — |
| 512×512 | 8× | 12× | — |
| 1024×1024 | **9×** | **20×** | 54× (dense spike, excluded) |

### Example (1024×1024):
```
Pattern   Dense(ms)   Sparse(ms)   Speedup
1:4       12618.09    670.56       18.82×
2:4       14662.58    1626.62      9.01×
2:8       13843.85    769.59       17.99×
4:16      10886.07    544.07       20.01×
```

---

## 🧬 Architecture
```
  PyTorch / ONNX
         │
         ▼
  MLIR Frontend
         │
         ▼
 SPA (Sparsity Analysis)
         │
         ▼
Rewrite Pass → sparse_matmul_N_M()
         │
         ▼
LLVM → CPU Runtime (OpenMP)
```

---

## 🔧 Quick Start
```bash
git clone https://github.com/MapleSilicon/SparseFlow
cd SparseFlow/compiler
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/lib/llvm-19 ..
make -j8
```

Run benchmarks:
```bash
cd ../../runtime/build
./benchmark_nm_runtime
```

---

## 🔬 Technical Highlights

### **SPA (Sparsity Propagation Analysis)**
* 2D sparsity (row + column)
* Determines which modes are sparse
* Zero-propagation tracking
* No runtime overhead

### **Rewrite Pass**
Automatically converts:
```mlir
linalg.matmul → func.call @sparse_matmul_N_M
```

### **Runtime**
Five specialized kernels:
* sparse_matmul_1_4
* sparse_matmul_2_4
* sparse_matmul_2_8
* sparse_matmul_4_16
* sparse_matmul_8_32

---

## 🛣 Roadmap (Updated)

### **v0.3 — Q1 2026**
* Begin GPU acceleration
* Initial CUDA kernels
* Tensor Core 2:4 prototype

### **v0.4 — Q2 2026**
* PyTorch integration
* torch.compile backend
* Python bindings

### **v0.5 — Q3 2026**
* Cloud pilot integration
* Production stabilization
* End-to-end deployment

---

## 🧪 Benchmarks

The full detailed benchmark suite is available in:
```bash
./runtime/benchmark_nm_runtime
```

---

## 🤝 Contributing

We welcome contributors in:
* MLIR / Compiler passes
* Sparse kernel optimization
* GPU kernel development
* PyTorch frontend
* Benchmarks

---

## 📫 Contact

**Email:** [maplesilicon1@gmail.com](mailto:maplesilicon1@gmail.com)  
**GitHub:** [https://github.com/MapleSilicon/SparseFlow](https://github.com/MapleSilicon/SparseFlow)

---

## 🆕 What's New in v0.2.0

* ✔️ Generalized N:M sparsity support
* ✔️ Runtime kernels for 1:4, 2:4, 2:8, 4:16, 8:32
* ✔️ Full benchmark suite
* ✔️ Updated SPA + Rewrite integration
* ✔️ New documentation and performance tables
