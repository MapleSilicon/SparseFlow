# SparseFlow Compiler Tests

## Purpose
Tests validate **compiler transformations**, not numerical correctness.

---

## Test Categories

### SPA Tests
- Validate N:M propagation

### Rewrite Tests
- Validate matmul → func.call lowering

### GPU Tests
- Validate:
  - gpu.module creation
  - gpu.func insertion
  - gpu.launch lowering
  - Verifier correctness

---

## Important
GPU tests do NOT validate computation.
They validate **legal IR construction only**.
