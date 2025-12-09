# SparseFlow v0.2 N:M Sparsity API

**Version:** 0.2.0  
**Date:** December 2025

## Overview

SparseFlow v0.2 introduces generalized N:M structured sparsity support, allowing you to work with any N:M pattern including 1:4, 2:4, 2:8, 4:16, and 8:32.

---

## MLIR API

### Annotating Operations with N:M Patterns

Use the `sparseflow.mark` operation to annotate tensors with N:M patterns:
```mlir
// 2:4 pattern (50% density)
%sparse = "sparseflow.mark"(%input) {
  n = 2 : i32,
  m = 4 : i32,
  direction = "row"
} : (tensor<128x128xf32>) -> tensor<128x128xf32>

// 1:4 pattern (25% density)
%sparse = "sparseflow.mark"(%input) {
  n = 1 : i32,
  m = 4 : i32,
  direction = "row"
} : (tensor<128x128xf32>) -> tensor<128x128xf32>
```

### Attributes

- `n` (IntegerAttr): Number of non-zero elements per block
- `m` (IntegerAttr): Block size
- `direction` (StringAttr): "row" or "col" (optional, defaults to "row")

---

## Compiler Passes

### SPA Pass (`sparseflow-spa`)

Analyzes and propagates N:M sparsity patterns through operations.

**Usage:**
```bash
mlir-opt input.mlir --sparseflow-spa -o analyzed.mlir
```

**Output Attributes:**
- `sparseflow.nm_n` - N value
- `sparseflow.nm_m` - M value
- `sparseflow.nm_direction` - Pattern direction
- `sparseflow.nm_pattern` - Human-readable name (e.g., "2:4")

### Rewrite Pass (`sparseflow-rewrite-matmul`)

Converts annotated matmul operations to sparse runtime calls.

**Usage:**
```bash
mlir-opt analyzed.mlir --sparseflow-rewrite-matmul -o rewritten.mlir
```

**Generated Functions:**
- `sparse_matmul_1_4` for 1:4 patterns
- `sparse_matmul_2_4` for 2:4 patterns
- `sparse_matmul_2_8` for 2:8 patterns
- `sparse_matmul_4_16` for 4:16 patterns
- `sparse_matmul_8_32` for 8:32 patterns

### Export Pass (`sparseflow-spa-export`)

Exports sparsity metadata to JSON.

**Usage:**
```bash
mlir-opt analyzed.mlir --sparseflow-spa-export
```

**Output:** `spa_sparsity.json`

**Format:**
```json
{
  "version": "v0.2",
  "operations": [
    {
      "name": "linalg.matmul",
      "nm_pattern": {
        "N": 2,
        "M": 4,
        "density": 0.5,
        "direction": "row",
        "pattern_name": "2:4"
      }
    }
  ]
}
```

---

## Runtime API

### C/C++ Functions

All runtime functions are in `sparseflow_runtime.h`:
```c
// 1:4 sparse matmul (25% density)
void sparse_matmul_1_4(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape);

// 2:4 sparse matmul (50% density)
void sparse_matmul_2_4(
    float* out, const float* lhs, const float* rhs,
    int64_t* out_shape, int64_t* lhs_shape, int64_t* rhs_shape);

// 2:8, 4:16, 8:32 also available
```

### Pattern Validation
```c
// Quick validation
bool validate_nm_pattern(
    const float* tensor,
    int rows, int cols,
    int N, int M,
    float zero_threshold);

// Detailed validation with statistics
ValidationResult validate_nm_pattern_detailed(
    const float* tensor,
    int rows, int cols,
    int N, int M,
    float zero_threshold);
```

---

## Example: Complete Pipeline
```bash
# 1. Annotate your MLIR with N:M patterns
cat > input.mlir << 'MLIR'
func.func @sparse_example(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = tensor.empty() : tensor<128x128xf32>
  %sparse_A = "sparseflow.mark"(%A) {n = 2 : i32, m = 4 : i32} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  %result = linalg.matmul ins(%sparse_A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                          outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}
MLIR

# 2. Run SPA analysis
mlir-opt-19 input.mlir \
  --load-pass-plugin=passes/SparseFlowPasses.so \
  --pass-pipeline="builtin.module(sparseflow-spa)" \
  -o analyzed.mlir

# 3. Rewrite to sparse kernels
mlir-opt-19 analyzed.mlir \
  --load-pass-plugin=passes/SparseFlowPasses.so \
  --pass-pipeline="func.func(sparseflow-rewrite-matmul)" \
  -o rewritten.mlir

# 4. Export metadata
mlir-opt-19 rewritten.mlir \
  --load-pass-plugin=passes/SparseFlowPasses.so \
  --sparseflow-spa-export
```

---

## Supported Patterns

| Pattern | N | M | Density | Use Case |
|---------|---|---|---------|----------|
| 1:4 | 1 | 4 | 25% | Highly sparse networks |
| 2:4 | 2 | 4 | 50% | NVIDIA Tensor Cores |
| 2:8 | 2 | 8 | 25% | Large sparse models |
| 4:16 | 4 | 16 | 25% | Memory-constrained |
| 8:32 | 8 | 32 | 25% | Very large models |

---

## Performance Tips

1. **Choose the right pattern**: Higher sparsity (lower density) = more speedup potential
2. **Align matrix dimensions**: Multiples of M give best performance
3. **Use OpenMP**: Set `OMP_NUM_THREADS` for multi-core speedup
4. **Validate in debug builds**: Use validation functions during development

---

## Migration from v0.1

### What Changed

- **v0.1**: Only 2:4 pattern supported
- **v0.2**: Any N:M pattern supported

### Code Changes

**v0.1:**
```mlir
// Hardcoded 2:4
%result = linalg.matmul ins(%A, %B) outs(%C)
```

**v0.2:**
```mlir
// Explicit N:M annotation
%sparse_A = "sparseflow.mark"(%A) {n = 2 : i32, m = 4 : i32}
%result = linalg.matmul ins(%sparse_A, %B) outs(%C)
```

### API Compatibility

✅ All v0.1 runtime functions still work  
✅ v0.1 pass names unchanged  
✅ JSON format is backwards compatible (has `version` field)

---

## Troubleshooting

### Pattern Not Detected

**Problem:** SPA doesn't detect your pattern

**Solution:** Ensure `sparseflow.mark` has correct `n` and `m` attributes

### Wrong Runtime Function Called

**Problem:** Expecting `sparse_matmul_1_4` but got `sparse_matmul_2_4`

**Solution:** Check SPA pass ran before rewrite pass

### Validation Fails

**Problem:** `validate_nm_pattern()` returns false

**Solution:** Check your tensor actually has N:M structure. Use `validate_nm_pattern_detailed()` for diagnostics.

---

## References

- [SparseFlow v0.1 Docs](../v0.1/)
- [ROADMAP.md](../../ROADMAP.md)
- [Implementation Plan](NM_SPARSITY_PLAN.md)

---

*Last Updated: December 2025*
