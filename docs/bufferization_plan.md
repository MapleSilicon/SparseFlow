# SparseFlow Bufferization Plan (v0.3-alpha)

This document defines how SparseFlow will safely transition from tensor IR to memref IR
without losing SparseFlow semantics.

## Goal
Make the pipeline stable under MLIR 19:
- `one-shot-bufferize`
- `finalizing-bufferize`
- GPU rewrite + launch operand plumbing

## Non-Goals (v0.3-alpha)
- No custom memref layout maps
- No custom memory spaces
- No mask materialization required yet
- No lowering to NVVM/ROCDL/SPIR-V required

## Contract: What must survive bufferization
SparseFlow currently relies on **operation-level attributes** (e.g., SPA masks attached to ops).
That is the correct choice for v0.3-alpha because:
- Tensor-type dictionary attributes may be dropped or transformed during bufferization.
- Operation attributes are stable across tensor→memref conversion as long as the op remains.

### Rule 1: Keep SparseFlow metadata on ops, not types (for now)
- SPA produces attributes like:
  - `sparseflow.spa_rowmask`
  - `sparseflow.spa_colmask`
- These must remain attached to the op(s) that the rewrite consumes (matmul or call).

### Rule 2: Insert/consume GPU constructs before heavy lowering
- `sparseflow-gpu-rewrite` should run before deep GPU lowering.
- In v0.3-alpha it may run:
  - before bufferization (current)
  - or after bufferization (future)
but it must not depend on tensor-type attrs.

## Recommended v0.3-alpha Pipeline (stable)
Minimal validated pipeline:

1. `sparseflow-spa`
2. `sparseflow-rewrite-matmul`
3. `sparseflow-gpu-rewrite`

Optional later:
4. `one-shot-bufferize{bufferize-function-boundaries=true}`
5. `func.func(finalizing-bufferize)`

## Planned v0.3-beta Evolution (when ABI becomes real)
Once we move from stubs to real kernel calls:

### Step A: Replace tensors with buffers at the call boundary
- Convert `func.call @sparse_matmul_*` or the original matmul into a call whose operands are memrefs.
- Ensure operands are explicit buffers (no implicit conversions inside the kernel).

### Step B: Materialize rowMask as a real buffer
- Decide format (bitmask vs index list)
- Materialize as:
  - constant global, OR
  - host-side alloc + init, OR
  - device-side init kernel
v0.3-alpha does not commit to which.

### Step C: GPU launch must consume the call (no dead stubs)
- `sparseflow-gpu-rewrite` must either:
  1) replace `func.call` with GPU launch + kernel call, or
  2) replace `linalg.matmul` directly (later).

## Why we are NOT using layout maps / memory spaces yet
Those decisions are high-risk early:
- They require downstream lowering agreement across multiple backends.
- They create debugging complexity before correctness is proven.

We only introduce them after:
- call-consumption is correct
- bufferized IR is stable
- kernel ABI is locked
