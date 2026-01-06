# SparseFlow Architecture

## Overview

SparseFlow is a **compiler-first** approach to sparse inference. Unlike libraries that provide hand-tuned kernels, SparseFlow uses MLIR to automatically optimize and fuse operations.
```
User Code (PyTorch)
        ↓
Python API (sparseflow)
        ↓
C++ Runtime (context, kernels)
        ↓
MLIR Compiler (optimization passes)
        ↓
Code Generator (CUDA kernels)
        ↓
GPU Execution (Ampere+ Tensor Cores)
```

---

## Key Components

### 1. Epilogue ABI (Week 1)

**Problem:** Every activation needs its own kernel.

**Solution:** Configurable epilogue system.
```cpp
enum class EpilogueKind {
    NONE,
    RELU,
    SILU,
    GELU,
    BIAS,
    BIAS_RELU,
    // ... extensible
};
```

**Benefits:**
- Single kernel handles any epilogue
- Easy to add new activations
- Compiler decides when to fuse

---

### 2. MLIR Optimization Passes (Week 2)

**Problem:** Manual tuning is brittle and slow.

**Solution:** Compiler automatically optimizes.

#### Pass 1: Tile Size Optimization
```mlir
// Before
%0 = sparseflow.sparse_gemm %A, %Bc : ...

// After
%0 = sparseflow.sparse_gemm %A, %Bc {
  tile_m = 128,  // Auto-selected
  tile_n = 128,  // for GPU arch
  tile_k = 64
} : ...
```

#### Pass 2: Operation Fusion
```mlir
// Before
%0 = sparseflow.sparse_gemm %A, %Bc
%1 = arith.silu %0

// After (single kernel)
%0 = sparseflow.fused_sparse_gemm %A, %Bc {
  epilogue = "silu"
}
```

**Benefits:**
- No manual tuning needed
- Adapts to different GPUs
- Finds optimizations humans miss

---

### 3. Python Integration (Week 3)

**Design Principles:**
1. **Explicit, not implicit** - No hidden behavior
2. **Auditable** - Show accuracy impact
3. **PyTorch native** - Feels like native PyTorch
```python
# Explicit conversion
sparse, diff = sf.SparseLinear.from_dense(
    dense,
    return_diff=True  # Show impact
)
print(f"Max error: {diff['max_error']}")
```

---

### 4. Stable ABI (Week 4)

**Problem:** Binary compatibility across versions.

**Solution:** C API with opaque handles.
```c
// Opaque handles (hide implementation)
typedef struct SparseFlowContext_* SparseFlowContext;
typedef struct SparseFlowKernel_* SparseFlowKernel;

// Versioned API
#define SPARSEFLOW_ABI_VERSION_MAJOR 1
#define SPARSEFLOW_ABI_VERSION_MINOR 0
```

**Benefits:**
- Upgrade internals without breaking users
- Enterprise requirement
- Thread-safe error handling

---

### 5. Deployment Tools (Week 5)

**Philosophy:** Prove ROI before deployment.

**Tools:**
1. **sparseflow-audit** - Cost/GPU calculator
2. **sparseflow-convert** - Model converter
3. **sparseflow-benchmark** - Performance validator

---

## Data Flow

### Compression Pipeline
```
Dense Weight (M×N, FP16)
        ↓
Prune to 2:4 (magnitude-based)
        ↓
Validate Pattern
        ↓
Compress (50% size)
        ↓
Sparse Weight (M×N/2, FP16)
```

### Inference Pipeline
```
Input (M×K dense)
        ↓
Sparse GEMM (with epilogue)
        ↓
Output (M×N dense)
```

**Key insight:** Only weights are sparse, activations stay dense.

---

## 2:4 Structured Sparsity

### What is 2:4?

For every 4 consecutive values, exactly 2 are zero:
```
Dense:   [1.2, 0.8, 0.3, 0.9]
2:4:     [1.2, 0.0, 0.0, 0.9]  ✓ Valid (2 zeros)
         [1.2, 0.8, 0.0, 0.0]  ✓ Valid (2 zeros)
         [0.0, 0.0, 0.0, 0.9]  ✗ Invalid (3 zeros)
```

### Why 2:4?

**Hardware support:** NVIDIA Ampere+ Tensor Cores have native 2:4 sparse operations.

**Performance:** 2× speedup (50% less data to move).

**Accuracy:** Minimal impact (< 1% on most tasks).

---

## Memory Layout

### Dense Layout
```
[w0, w1, w2, w3, w4, w5, w6, w7, ...]
```

### 2:4 Compressed Layout
```
Values:   [w0, w3, w4, w7, ...]  (only non-zero)
Metadata: [0b1001, 0b1001, ...]  (4-bit masks)
```

**Compression:** 50% size reduction

---

## Performance Model

### Theoretical Speedup
```
Speedup = (Dense FLOPS) / (Sparse FLOPS)
        = 1.0 / 0.5  (50% of weights)
        = 2.0×
```

### Real-World Speedup

Factors:
- **Memory bandwidth** (sparse uses 50% less)
- **Tensor Core utilization** (hardware support)
- **Batch size** (need ≥32 for full speedup)
- **Matrix size** (larger = better)

**Typical:** 1.8-2.1× on Ampere GPUs

---

## Compiler Optimizations

### 1. Tile Size Selection

Based on:
- GPU architecture (SM count, registers, shared memory)
- Matrix dimensions
- Occupancy analysis
```python
def select_tile_size(M, N, K, gpu_arch):
    candidates = [(64,64,32), (128,128,64), ...]
    
    best = max(candidates, key=lambda t: 
        occupancy(t, gpu_arch) * work_per_thread(t)
    )
    
    return best
```

### 2. Epilogue Fusion

**Pattern matching:**
```mlir
gemm + relu        → fused_gemm(epilogue=relu)
gemm + bias + silu → fused_gemm(epilogue=bias_silu)
```

**Benefit:** Single kernel launch, no intermediate storage.

---

## Extensibility

### Adding New Epilogues

1. Add to enum:
```cpp
enum class EpilogueKind {
    // ...
    SWISH = 7,  // New!
};
```

2. Add code generator:
```cpp
case EpilogueKind::SWISH:
    return "return __hmul(x, sigmoid(x));";
```

3. Add MLIR fusion pattern:
```cpp
struct FuseGemmSwish : public OpRewritePattern<SwishOp> {
    // Pattern matching logic
};
```

**No kernel rewrite needed!**

---

## Future Directions

### Short Term
- INT8 support (4× speedup)
- More epilogue patterns
- Multi-GPU scaling

### Long Term
- Other sparsity patterns (4:8, 8:16)
- Dynamic sparsity
- Sparse training

---

## Design Decisions

### Why MLIR?

**Alternatives considered:**
- Hand-coded kernels → Not extensible
- Template metaprogramming → Complex, brittle
- JIT compilation → Slow first run

**MLIR advantages:**
- Composable passes
- Multiple backends
- Industry standard (used by TensorFlow, PyTorch)

### Why Stable C ABI?

**Enterprise requirement:** Binary compatibility across versions.

**Alternative:** C++ API → Template changes break ABI

**Solution:** C API with opaque handles

### Why Explicit Python API?

**User feedback:** Developers hate "magic" that breaks silently.

**Design:** Show every step, report every impact.
```python
# Bad (implicit)
model = sf.sparsify(model)  # What changed?

# Good (explicit)
sparse, diff = sf.SparseLinear.from_dense(dense, return_diff=True)
print(f"Impact: {diff['max_error']}")  # User knows!
```

---

## References

- [NVIDIA Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [Structured Sparsity Paper](https://arxiv.org/abs/2104.08378)
- [MLIR Documentation](https://mlir.llvm.org/)

---

**Built for engineers who need to understand their tools.**
