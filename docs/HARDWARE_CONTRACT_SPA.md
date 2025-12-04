# SparseFlow SPA Hardware Contract (v0.1)

## 1. Purpose

This document defines the **hardware interface** for a sparse matmul accelerator
that consumes row/column sparsity information produced by SparseFlow's
Sparsity Propagation Analysis (SPA) and accelerates matrix multiplication.

SPA gives us:
- `rowmask`  — which rows of A are active (non-zero)
- `colmask`  — which columns of B are active (non-zero)
- Expected FLOP reduction (e.g., 75% at 50% row + 50% col sparsity)

The accelerator is expected to:
- Skip masked rows/columns
- Compute only on active blocks
- Provide a **3–4× speedup** vs dense CPU baseline at ~75% sparsity

---

## 2. Mathematical Operation

We compute:

> C = A × B

Where:
- A is shape `[M × K]`
- B is shape `[K × N]`
- C is shape `[M × N]`

All tensors are **row-major**, **32-bit float** (`float32`).

Sparsity:

- `rowmask[i] = 0` → entire row `i` of A is structurally zero → skip
- `colmask[j] = 0` → entire column `j` of B is structurally zero → skip

Correctness requirement:
- For all active `(i, j)` where `rowmask[i] == 1` and `colmask[j] == 1`,
  `C_hw[i, j]` must match dense matmul within numerical tolerance:

> |C_hw[i, j] − C_dense[i, j]| ≤ 1e−4 (for typical ranges)

Masked rows/cols in C may be:
- Either zero-filled, or
- Left undefined (but currently we expect **zero** for simplicity)

---

## 3. Data Layout & Types

All tensors are **contiguous row-major**:

- `A`:
  - Type: `float32`
  - Shape: `[M × K]`
  - Layout: `A[i, k]` at `A_base + (i * K + k) * 4` bytes

- `B`:
  - Type: `float32`
  - Shape: `[K × N]`
  - Layout: `B[k, j]` at `B_base + (k * N + j) * 4` bytes

- `C`:
  - Type: `float32`
  - Shape: `[M × N]`
  - Layout: `C[i, j]` at `C_base + (i * N + j) * 4` bytes

Masks:

- `rowmask`:
  - Type: `uint8_t`
  - Length: `M`
  - Semantics: `rowmask[i] ∈ {0,1}`

- `colmask`:
  - Type: `uint8_t`
  - Length: `N`
  - Semantics: `colmask[j] ∈ {0,1}`

No compression (no bit-packing) in v0.1. Simple and dumb on purpose.

---

## 4. C / Runtime API Contract

### 4.1 Baseline CPU Reference (Already Implemented)

The current CPU reference kernel has the signature:

```c
// CPU reference version (OpenMP, blocked, used today)
void sparseflow_masked_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    const uint8_t* rowmask,
    const uint8_t* colmask,
    unsigned long& flops);



