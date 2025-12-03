# SPA API Reference

## Pass Registration

### `sparseflow-spa`

**Type:** `OperationPass<ModuleOp>`

**Description:** Performs sparsity propagation analysis on MLIR module.

**Usage:**
```bash
--pass-pipeline='builtin.module(sparseflow-spa)'
```

**Options:** None

## Attributes

### `sparseflow.n`

**Type:** `IntegerAttr` (i32)

**Attached to:** Operations (typically `linalg.matmul`)

**Meaning:** Numerator in N:M sparsity pattern (non-zero values)

**Example:**
```mlir
{sparseflow.n = 2 : i32}
```

### `sparseflow.m`

**Type:** `IntegerAttr` (i32)

**Attached to:** Operations (typically `linalg.matmul`)

**Meaning:** Denominator in N:M sparsity pattern (total values)

**Example:**
```mlir
{sparseflow.m = 4 : i32}
```

### `sparseflow.spa_rowmask`

**Type:** `ArrayAttr` of `BoolAttr`

**Attached to:** Operations with tensor results

**Meaning:** Per-row sparsity information
- `true` = row may contain non-zero values
- `false` = row is provably zero

**Example:**
```mlir
{sparseflow.spa_rowmask = [true, true, false, false]}
```

Interpretation: Rows 0 and 1 may be non-zero, rows 2 and 3 are zero.

### `sparseflow.spa_colmask`

**Type:** `ArrayAttr` of `BoolAttr`

**Attached to:** Operations with tensor results (v0.6+)

**Meaning:** Per-column sparsity information

**Status:** Placeholder in v0.5, full support in v0.6

## C++ API

### Data Structures

#### `MatrixSparsity`
```cpp
namespace sparseflow {
  struct MatrixSparsity {
    std::vector<std::uint8_t> rowMask;
    std::vector<std::uint8_t> colMask;  // v0.6+
  };
}
```

**Fields:**
- `rowMask`: 1 = maybe non-zero, 0 = proven zero
- `colMask`: (future) column-level sparsity

### Functions

#### `makeDenseRows`
```cpp
MatrixSparsity makeDenseRows(int rows);
```

**Description:** Create a fully-dense (all-ones) row mask.

**Parameters:**
- `rows`: Number of rows

**Returns:** MatrixSparsity with all rows marked non-zero

**Example:**
```cpp
auto dense = makeDenseRows(4);
// dense.rowMask = [1, 1, 1, 1]
```

#### `makeDenseRowsCols`
```cpp
MatrixSparsity makeDenseRowsCols(int rows, int cols);
```

**Description:** Create fully-dense row and column masks.

**Parameters:**
- `rows`: Number of rows
- `cols`: Number of columns

**Returns:** MatrixSparsity with all rows and columns marked non-zero

#### `intersectRows`
```cpp
MatrixSparsity intersectRows(const MatrixSparsity &a, 
                              const MatrixSparsity &b);
```

**Description:** Compute AND of two sparsity patterns (for multiplication).

**Parameters:**
- `a`: First sparsity pattern
- `b`: Second sparsity pattern

**Returns:** Intersection pattern (AND operation)

**Semantics:**
```
out.rowMask[i] = a.rowMask[i] & b.rowMask[i]
```

**Example:**
```cpp
// a.rowMask = [1, 1, 0, 1]
// b.rowMask = [1, 0, 1, 1]
auto c = intersectRows(a, b);
// c.rowMask = [1, 0, 0, 1]  (AND)
```

#### `unionRows`
```cpp
MatrixSparsity unionRows(const MatrixSparsity &a,
                         const MatrixSparsity &b);
```

**Description:** Compute OR of two sparsity patterns (for addition).

**Parameters:**
- `a`: First sparsity pattern
- `b`: Second sparsity pattern

**Returns:** Union pattern (OR operation)

**Semantics:**
```
out.rowMask[i] = a.rowMask[i] | b.rowMask[i]
```

**Example:**
```cpp
// a.rowMask = [1, 1, 0, 0]
// b.rowMask = [1, 0, 1, 0]
auto c = unionRows(a, b);
// c.rowMask = [1, 1, 1, 0]  (OR)
```

### Pass Class

#### `SparsityPropagationPass`
```cpp
struct SparsityPropagationPass 
    : public PassWrapper<SparsityPropagationPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final;
  StringRef getDescription() const final;
  void runOnOperation() override;
};
```

**Key Methods:**

##### `getSparsity`
```cpp
MatrixSparsity getSparsity(Value v, int64_t expectedRows);
```

**Description:** Retrieve sparsity information for an SSA value.

**Parameters:**
- `v`: MLIR value to query
- `expectedRows`: Expected number of rows (for normalization)

**Returns:** Sparsity pattern for the value

**Behavior:**
1. Check internal SparsityMap cache
2. Check for `spa_rowmask` attribute on defining op
3. Check for N:M attributes
4. Default to dense if no info found

##### `attachRowMaskAttr`
```cpp
void attachRowMaskAttr(Operation *op, const MatrixSparsity &info);
```

**Description:** Attach `spa_rowmask` attribute to operation.

**Parameters:**
- `op`: Operation to annotate
- `info`: Sparsity pattern to attach

**Effect:** Sets `sparseflow.spa_rowmask` attribute on operation

## Reading SPA Results in Your Pass

### Example: Query Sparsity
```cpp
#include "mlir/IR/Attributes.h"

void myOptimizationPass(Operation *op) {
  // Check if operation has sparsity info
  if (auto rowmask = op->getAttrOfType<ArrayAttr>("sparseflow.spa_rowmask")) {
    
    // Iterate over rows
    for (size_t i = 0; i < rowmask.size(); ++i) {
      bool isNonZero = rowmask[i].cast<BoolAttr>().getValue();
      
      if (!isNonZero) {
        // Row i is provably zero - skip computation
        optimizeZeroRow(op, i);
      }
    }
  }
}
```

### Example: Count Zero Rows
```cpp
size_t countZeroRows(Operation *op) {
  auto rowmask = op->getAttrOfType<ArrayAttr>("sparseflow.spa_rowmask");
  if (!rowmask) return 0;
  
  size_t zeros = 0;
  for (auto attr : rowmask) {
    if (!attr.cast<BoolAttr>().getValue()) {
      zeros++;
    }
  }
  return zeros;
}
```

### Example: Check N:M Pattern
```cpp
std::optional<std::pair<int, int>> getNMPattern(Operation *op) {
  auto nAttr = op->getAttrOfType<IntegerAttr>("sparseflow.n");
  auto mAttr = op->getAttrOfType<IntegerAttr>("sparseflow.m");
  
  if (nAttr && mAttr) {
    return std::make_pair(nAttr.getInt(), mAttr.getInt());
  }
  return std::nullopt;
}
```

## Type Definitions

### `SparsityMap`
```cpp
using SparsityMap = llvm::DenseMap<mlir::Value, MatrixSparsity>;
```

**Description:** Internal map from SSA values to their sparsity patterns.

**Usage:** Maintained by SPA pass during analysis.

## Future API (v0.6+)

### `intersect2D` / `union2D`
```cpp
MatrixSparsity intersect2D(const MatrixSparsity &a, const MatrixSparsity &b);
MatrixSparsity union2D(const MatrixSparsity &a, const MatrixSparsity &b);
```

**Description:** 2D operations on both row and column masks.

### `transposeSparsity`
```cpp
MatrixSparsity transposeSparsity(const MatrixSparsity &m);
```

**Description:** Swap row and column masks for transpose operation.

