# SPA Architecture

## System Design

SPA is implemented as an MLIR pass that operates on function-level IR. It performs a single forward walk of the computation graph, building a sparsity map as it goes.

## Core Components

### 1. Sparsity Domain (`SPADomain.h/cpp`)

Defines the abstract sparsity representation:
```cpp
struct MatrixSparsity {
  std::vector<uint8_t> rowMask;  // 1 = maybe non-zero, 0 = proven zero
  std::vector<uint8_t> colMask;  // (v0.6+)
};
```

**Operations:**
- `makeDenseRows(rows)` - Create all-ones mask
- `intersect2D(a, b)` - AND operation (for mul)
- `union2D(a, b)` - OR operation (for add)

### 2. Analysis Pass (`SparsityPropagationPass.cpp`)

Main pass that:
1. Walks the IR operation-by-operation
2. Computes output sparsity from input sparsity
3. Attaches `sparseflow.spa_rowmask` attributes

**Key methods:**
- `getSparsity(value, expectedRows)` - Retrieve sparsity info
- `handleMatmul()` - Matmul-specific propagation
- `handleAddLike()` - Union semantics (add, sub)
- `handleMulLike()` - Intersection semantics (mul, div)

## Data Flow
```
┌─────────────────────────────────────┐
│  Input MLIR (with N:M attributes)   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  SPA Pass - Forward Walk             │
│                                      │
│  SparsityMap: Value → MatrixSparsity│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  For each operation:                 │
│  1. Read input sparsity              │
│  2. Apply propagation rule           │
│  3. Store output sparsity            │
│  4. Attach attribute                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Output MLIR (with rowmask attrs)   │
└─────────────────────────────────────┘
```

## Propagation Rules

### Matmul: `C = A × B`

**Current (v0.5 - Row-only):**
```
rowMask(C)[i] = rowMask(A)[i]
```

**Future (v0.6 - 2D):**
```
rowMask(C)[i] = rowMask(A)[i] AND (∃j: colMask(B)[j])
colMask(C)[j] = colMask(B)[j] AND (∃i: rowMask(A)[i])
```

### Element-wise Add/Sub: `C = A + B`
```
rowMask(C)[i] = rowMask(A)[i] OR rowMask(B)[i]
```

Row is zero only if **both** inputs have zero row.

### Element-wise Mul/Div: `C = A * B`
```
rowMask(C)[i] = rowMask(A)[i] AND rowMask(B)[i]
```

Row is zero if **either** input has zero row.

### Maximum (ReLU): `C = max(A, 0)`
```
rowMask(C)[i] = rowMask(A)[i]
```

Zero rows stay zero.

## N:M Integration

When `sparseflow.n` and `sparseflow.m` attributes exist:

1. **Detection**: Pass reads N and M integers
2. **Pattern Generation**: Creates repeating pattern
   - Example: N=2, M=4 → `[T, T, F, F, T, T, F, F, ...]`
3. **Storage**: Stored in SparsityMap for propagation
```
┌──────────────────────┐
│  AnnotateNm Pass     │
│  Adds: n=2, m=4      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  SPA detects N:M     │
│  Pattern: [T,T,F,F]  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Propagates to ops   │
│  downstream          │
└──────────────────────┘
```

## Extension Points

To add support for new operations:

1. Add handler method (e.g., `handleConv()`)
2. Register in `runOnOperation()` walk
3. Define propagation semantics
4. Add test case

Example:
```cpp
void handleConv(linalg::ConvOp conv) {
  // Read input sparsity
  MatrixSparsity input = getSparsity(conv.getInput(), ...);
  
  // Apply conv-specific logic
  MatrixSparsity output = ...;
  
  // Store and attach
  S[conv.getResult()] = output;
  attachRowMaskAttr(conv, output);
}
```

## Performance

- **Analysis Time**: O(n) where n = number of operations
- **Memory**: O(v) where v = number of SSA values
- **No runtime overhead**: All analysis is compile-time

## Future Directions

1. **Inter-procedural analysis** - Track sparsity across function calls
2. **Tile-level granularity** - 16×16 or 32×32 blocks
3. **Dynamic profiling** - Runtime feedback to compiler
4. **Auto-kernel selection** - Choose sparse kernels based on SPA info

