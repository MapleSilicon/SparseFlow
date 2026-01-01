# 🔒 SparseFlow GPU Kernel ABI — v0.3-alpha (FROZEN)

This ABI defines the **contract between the compiler and GPU kernels**.
Everything above it (MLIR passes) and below it (CUDA/HIP/LLVM lowering) must obey this.

---

## 0️⃣ Design Principles (Non-Negotiable)

1. **Explicit memory, no tensors** — GPU kernels only see `memref` (raw pointers)
2. **No implicit metadata** — All sparsity info is passed explicitly
3. **Stable across hardware** — Same ABI works for CUDA, ROCm, CPU SIMD later
4. **Host owns setup** — GPU kernel only computes
5. **Structured N:M sparsity only** — No unstructured masks in v0.x

---

## 1️⃣ Kernel Naming Convention (Frozen)
```
sparseflow_gpu_matmul_<N>_<M>_<dtype>
```

Examples:
- `sparseflow_gpu_matmul_2_4_f32`
- `sparseflow_gpu_matmul_4_8_f16`

---

## 2️⃣ Kernel Signature (THE ABI)

### MLIR-level ABI (after bufferization)
```mlir
gpu.func @sparseflow_gpu_matmul_2_4_f32(
  %A        : memref<?x?xf32>,     // LHS matrix
  %B        : memref<?x?xf32>,     // RHS matrix
  %C        : memref<?x?xf32>,     // Output matrix
  %rowMask  : memref<?xi32>,       // Bitmask (1 bit per row)
  %M        : i32,                 // rows of A / C
  %N        : i32,                 // cols of B / C
  %K        : i32                  // reduction dim
) kernel
```

---

## 3️⃣ Argument Semantics (Frozen)

| Arg | Ownership | Description |
|-----|-----------|-------------|
| `A` | Read-only | Dense matrix with structured sparsity |
| `B` | Read-only | Fully dense |
| `C` | Write-only | Dense output |
| `rowMask` | Read-only | Bitmask encoding non-zero rows |
| `M` | Scalar | Number of rows |
| `N` | Scalar | Number of columns |
| `K` | Scalar | Reduction dimension |

❗ No aliasing allowed (`restrict` semantics).

---

## 4️⃣ rowMask ABI (Locked)

### Format: **Bitmask**
```
rowMask[i / 32] & (1 << (i % 32)) != 0
→ row i is ACTIVE
```

### Invariants (guaranteed by compiler):
- Exactly **N active rows per M rows**
- Pattern repeats every `M`
- Bounds checks NOT required in kernel
- rowMask length = `ceil(M / 32)`

This is **GPU-optimal** and **SIMD-friendly**.

---

## 5️⃣ Memory Spaces (Frozen)

| Buffer | Memory Space |
|--------|--------------|
| `A`, `B`, `C` | Global memory |
| `rowMask` | Global (read-only, cacheable) |
| Tile scratchpads (future) | Shared memory (space 3) |

No implicit shared memory in v0.3.

---

## 6️⃣ Launch Semantics (Compiler Responsibility)

The compiler guarantees:
```mlir
gpu.launch blocks(%bx, %by, %bz) threads(%tx, %ty, %tz)
```

- One warp (or subgroup) processes **one M-row block**
- Thread layout is **kernel-specific**, not ABI-specific
- ABI does NOT encode tile sizes (future-proof)

---

## 7️⃣ What the Kernel May Assume

✅ Assumptions kernel may rely on:
- Structured N:M sparsity
- rowMask correctness
- No aliasing
- Correct bounds

❌ Kernel must NOT assume:
- Specific block size
- Specific hardware
- Any tensor metadata
- Any MLIR-level attributes

---

## 8️⃣ Compiler Contract

Before calling the kernel, compiler must:
1. Bufferize tensors → memrefs
2. Allocate and initialize `rowMask`
3. Compute `M, N, K`
4. Emit `gpu.launch`
5. Never pass tensors to GPU

Failure = compiler bug, not kernel bug.

---

## 9️⃣ ABI Stability Promise

This ABI is:
- **Frozen for v0.3.x**
- **Binary compatible across v0.3.x**
- Will only change in **v0.4 (major GPU rev)**

Future versions may add:
- Column masks
- Block masks
- Mixed precision
- Tensor cores

But **this ABI remains valid**.

---

## 🔏 Canonical ABI Summary
```
Kernel: sparseflow_gpu_matmul_<N>_<M>_<dtype>

Args:
  A        memref<?x?xT>
  B        memref<?x?xT>
  C        memref<?x?xT>
  rowMask memref<?xi32>
  M, N, K i32

Sparsity: Structured N:M, Bitmask per row, Compiler-validated

No tensors. No magic. No excuses.
```

---

## Next Steps (v0.4)

1. ✅ ABI frozen (v0.3-alpha)
2. ⏭ Implement bufferization mapping to this ABI
3. ⏭ Generate rowMask on host
4. ⏭ Lower to LLVM / CUDA
5. ⏭ Write first real kernel
