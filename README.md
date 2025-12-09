# 🌲 SparseFlow v0.2.0  
### Generalized N:M Sparse Compiler for AI Inference (MLIR + CPU Runtime)

SparseFlow is a next-generation MLIR-based compiler that detects and exploits **generalized structured sparsity (N:M)** in AI workloads.

Unlike traditional sparse libraries (limited to 2:4 or fully unstructured), SparseFlow supports **any N:M block pattern** and achieves **massive CPU acceleration** using compile-time analysis + custom sparse kernels.

---

## 🚀 Key Features (v0.2.0)

### ✅ Generalized N:M Sparsity  
Supports the following patterns out of the box:

- 1:4  
- 2:4  
- 2:8  
- 4:16  
- 8:32  

### ✅ MLIR Compiler Integration  
- SPA Pass — Static sparsity analysis  
- Rewrite Pass — Converts dense matmuls → sparse kernels  
- Export Pass — Dumps metadata  
- Pluggable runtime lowering

### ✅ Optimized CPU Runtime  
- 5 hand-tuned OpenMP kernels  
- Contiguous block loads  
- Branch-free inner loops  
- High cache locality  
- Designed for future SIMD + GPU backend

### ✅ Real Performance  
SparseFlow achieves **9×–20× speedup** on CPU for realistic matrix sizes, significantly outperforming typical sparse CPU libraries.

---

## 📊 Benchmark Results (REAL HARDWARE)

Benchmarks compare dense vs SparseFlow sparse kernels on CPU.

| Matrix Size | Typical Speedup | Peak Speedup |
|-------------|------------------|----------------|
| **256×256** | 3×–8× | 8× |
| **512×512** | 8×–12× | 12× |
| **1024×1024** | 9×–20× | 20× |

Stable patterns frequently hit:

- **1:4 → ~18×**
- **2:8 → ~18×**
- **4:16 → ~20×**

These numbers are based on multiple runs and exclude outlier spikes.

---

## 🧪 Example Benchmark Output
```
Matrix Size: 1024×1024
┌─────────┬────────────┬────────────┬──────────┬───────────┐
│ Pattern │ Dense (ms) │ Sparse (ms)│ Speedup  │ Density   │
├─────────┼────────────┼────────────┼──────────┼───────────┤
│ 1:4     │ 12618.09   │ 670.56     │ 18.82×   │ 25%       │
│ 2:4     │ 14662.58   │ 1626.62    │ 9.01×    │ 50%       │
│ 2:8     │ 13843.85   │ 769.59     │ 17.99×   │ 25%       │
│ 4:16    │ 10886.07   │ 544.07     │ 20.01×   │ 25%       │
└─────────┴────────────┴────────────┴──────────┴───────────┘
```

---

## 🏗 Compiler Pipeline

SparseFlow transforms dense MLIR into sparse-optimized executable code:
```
PyTorch / ONNX → MLIR → SPA Pass → Rewrite Pass → LLVM → Sparse Runtime
```

### 1. SPA Pass  
Identifies sparse regions and marks tensors with `{n, m}` metadata.

### 2. Rewrite Pass  
Replaces `linalg.matmul` with:
```mlir
func.call @sparse_matmul_N_M(...)
```

Dynamically choosing the correct sparse kernel.

### 3. Runtime  
Backed by optimized C++/OpenMP kernels:
```cpp
sparse_matmul_1_4
sparse_matmul_2_4
sparse_matmul_2_8
sparse_matmul_4_16
sparse_matmul_8_32
```

---

## 🧩 Supported Sparsity Patterns

A pattern **N:M** means:

- For every M consecutive weights  
- Exactly N are non-zero  
- Zeros are static at compile time  
- Blocks are memory contiguous  

This allows:

- Predictable skipping  
- SIMD-friendly loads  
- Low branch divergence  
- Great cache efficiency  

---

## 🔬 Example MLIR Input
```mlir
%A = tensor<16x16xf32> {n = 2 : i32, m = 8 : i32}
%B = tensor<16x16xf32>
%C = tensor<16x16xf32>

%0 = linalg.matmul ins(%A, %B)
```

### After Rewrite Pass:
```mlir
func.call @sparse_matmul_2_8(%A, %B, %C, %m, %k, %n)
```

---

## 📦 Build Instructions
```bash
git clone https://github.com/MapleSilicon/SparseFlow
cd SparseFlow/compiler
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/lib/llvm-19 ..
make -j8
```

### Run benchmarks
```bash
cd ../../runtime/build
./benchmark_nm_runtime
```

---

## 🗺 Roadmap

### **v0.3 (Q1 2026) — GPU Acceleration**
* CUDA kernels
* Tensor Core support
* 30–60× expected speedup

### **v0.4 (Q2 2026) — PyTorch Integration**
* Python bindings
* `torch.compile` backend
* Model zoo support

### **v0.5 (Q3 2026) — Production Deployment**
* Cloud provider pilots
* Enterprise safety and tooling

---

## 🤝 Contact

**Email:** maplesilicon1@gmail.com  
**GitHub:** https://github.com/MapleSilicon/SparseFlow  
**Author:** Gourav Kumar

---

# 🌲 SparseFlow

**Generalized Sparse Compute for AI.**  
**Simple. Fast. Open.**
