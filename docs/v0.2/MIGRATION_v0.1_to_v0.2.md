# Migration Guide: v0.1 → v0.2

## Summary of Changes

SparseFlow v0.2 generalizes sparsity support from hardcoded 2:4 to arbitrary N:M patterns.

### Key Improvements

✅ **Support for 1:4, 2:4, 2:8, 4:16, 8:32 patterns**  
✅ **Pattern-aware propagation**  
✅ **Runtime validation**  
✅ **JSON metadata with pattern info**

---

## Breaking Changes

### None! 

v0.2 is **fully backwards compatible** with v0.1.

- All v0.1 code continues to work
- 2:4 is still the default pattern
- No API changes required

---

## New Features

### 1. Explicit Pattern Annotation

**Before (v0.1):**
```mlir
// Pattern was implicit
%result = linalg.matmul ins(%A, %B) outs(%C)
```

**After (v0.2):**
```mlir
// Explicit N:M annotation
%sparse = "sparseflow.mark"(%A) {n = 1 : i32, m = 4 : i32}
%result = linalg.matmul ins(%sparse, %B) outs(%C)
```

### 2. Runtime Function Selection

**v0.1:**
- Only `sparse_matmul_2_4` available

**v0.2:**
- `sparse_matmul_1_4`
- `sparse_matmul_2_4` (same as v0.1)
- `sparse_matmul_2_8`
- `sparse_matmul_4_16`
- `sparse_matmul_8_32`

### 3. Pattern Validation

**New in v0.2:**
```c
bool valid = validate_nm_pattern(tensor, rows, cols, N, M, 1e-6f);
```

---

## Recommended Upgrade Path

### Step 1: Test Existing Code

Your v0.1 code should work unchanged:
```bash
# Run your existing tests
./test_jit_correctness
./benchmark_suite
```

### Step 2: Add Pattern Annotations (Optional)

If you want to use patterns other than 2:4:
```mlir
// Add sparseflow.mark operations
%sparse_1_4 = "sparseflow.mark"(%input) {n = 1 : i32, m = 4 : i32}
```

### Step 3: Update Build (If Needed)

Recompile passes and runtime:
```bash
cd compiler/build && make -j4
cd ../../runtime/build && make -j4
```

### Step 4: Validate Performance

Run benchmarks with different patterns:
```bash
./benchmark_nm_patterns
```

---

## FAQ

**Q: Do I need to change my v0.1 code?**  
A: No, it works as-is.

**Q: How do I use 1:4 instead of 2:4?**  
A: Add `sparseflow.mark` with `{n = 1 : i32, m = 4 : i32}`

**Q: Will my v0.1 JSON exports still work?**  
A: Yes, v0.2 adds fields but doesn't remove any.

**Q: What about performance?**  
A: v0.2 is as fast or faster than v0.1 for 2:4 patterns.

---

## Support

- GitHub Issues: https://github.com/MapleSilicon/SparseFlow/issues
- Email: maplesilicon1@gmail.com

---

*Migration Guide v0.2 - December 2025*
