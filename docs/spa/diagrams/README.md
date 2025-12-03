# SPA Diagrams

Visual representations of SPA concepts and data flow.

## System Architecture
```
┌────────────────────────────────────────────────────────────┐
│                    SparseFlow Compiler                      │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │ AnnotateNm   │──────▶│  SPA Pass    │                   │
│  │ Pass         │ N:M   │              │                   │
│  │              │ attrs │              │                   │
│  └──────────────┘       └──────┬───────┘                   │
│                                 │ rowmasks                  │
│                                 ▼                           │
│                         ┌──────────────┐                    │
│                         │ Optimization │                    │
│                         │ Passes       │                    │
│                         └──────────────┘                    │
└────────────────────────────────────────────────────────────┘
```

## N:M to Rowmask Conversion

### Example: 2:4 Sparsity Pattern
```
Input: N=2, M=4 (2 out of 4 values non-zero = 50% sparse)

Pattern Generation:
┌───────────────────────────────┐
│ Row 0:  ✓ ✓ ✗ ✗  (i%4 < 2)   │ → true
│ Row 1:  ✓ ✓ ✗ ✗  (i%4 < 2)   │ → true  
│ Row 2:  ✗ ✗ ✗ ✗  (i%4 < 2)   │ → false
│ Row 3:  ✗ ✗ ✗ ✗  (i%4 < 2)   │ → false
└───────────────────────────────┘

Resulting rowmask: [true, true, false, false]
```

## Propagation Rules Visualization

### 1. Matmul (A × B → C)
```
Matrix A (4×4)          Matrix B (4×4)          Matrix C (4×4)
rowmask: [T,T,F,F]     (no mask yet)           rowmask: [T,T,F,F]
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│ ▓▓▓▓▓▓▓▓▓▓ │         │ ▓▓▓▓▓▓▓▓▓▓ │         │ ▓▓▓▓▓▓▓▓▓▓ │
│ ▓▓▓▓▓▓▓▓▓▓ │    ×    │ ▓▓▓▓▓▓▓▓▓▓ │    =    │ ▓▓▓▓▓▓▓▓▓▓ │
│ ············ │         │ ▓▓▓▓▓▓▓▓▓▓ │         │ ············ │
│ ············ │         │ ▓▓▓▓▓▓▓▓▓▓ │         │ ············ │
└─────────────┘         └─────────────┘         └─────────────┘
    ^                                                ^
    |                                                |
Zero rows in A  ────────────────────────▶  Zero rows in C
```

### 2. Element-wise Add (A + B → C)
```
Matrix A (4×4)          Matrix B (4×4)          Matrix C (4×4)
rowmask: [T,F,F,T]     rowmask: [F,T,F,T]     rowmask: [T,T,F,T]
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│ ▓▓▓▓▓▓▓▓▓▓ │         │ ············ │         │ ▓▓▓▓▓▓▓▓▓▓ │
│ ············ │    +    │ ▓▓▓▓▓▓▓▓▓▓ │    =    │ ▓▓▓▓▓▓▓▓▓▓ │
│ ············ │         │ ············ │         │ ············ │
│ ▓▓▓▓▓▓▓▓▓▓ │         │ ▓▓▓▓▓▓▓▓▓▓ │         │ ▓▓▓▓▓▓▓▓▓▓ │
└─────────────┘         └─────────────┘         └─────────────┘

Rule: OR operation (union)
Row is zero only if BOTH inputs have zero row
```

### 3. Element-wise Multiply (A × B → C)
```
Matrix A (4×4)          Matrix B (4×4)          Matrix C (4×4)
rowmask: [T,T,F,T]     rowmask: [T,F,T,T]     rowmask: [T,F,F,T]
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│ ▓▓▓▓▓▓▓▓▓▓ │         │ ▓▓▓▓▓▓▓▓▓▓ │         │ ▓▓▓▓▓▓▓▓▓▓ │
│ ▓▓▓▓▓▓▓▓▓▓ │    ×    │ ············ │    =    │ ············ │
│ ············ │         │ ▓▓▓▓▓▓▓▓▓▓ │         │ ············ │
│ ▓▓▓▓▓▓▓▓▓▓ │         │ ▓▓▓▓▓▓▓▓▓▓ │         │ ▓▓▓▓▓▓▓▓▓▓ │
└─────────────┘         └─────────────┘         └─────────────┘

Rule: AND operation (intersection)
Row is zero if EITHER input has zero row
```

## Complete Pipeline Example
```
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Input MLIR                                           │
├──────────────────────────────────────────────────────────────┤
│ func @test(%A, %B) {                                         │
│   %C = linalg.matmul ins(%A, %B)                            │
│   %D = arith.addf %C, %C                                     │
│   return %D                                                   │
│ }                                                             │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 2: AnnotateNm Pass                                      │
├──────────────────────────────────────────────────────────────┤
│ func @test(%A, %B) {                                         │
│   %C = linalg.matmul {n=2, m=4} ins(%A, %B)                │
│   %D = arith.addf %C, %C                                     │
│   return %D                                                   │
│ }                                                             │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 3: SPA Pass - Analyze & Propagate                      │
├──────────────────────────────────────────────────────────────┤
│ • Read N:M (2:4)                                             │
│ • Generate pattern: [T, T, F, F]                            │
│ • Attach to matmul                                           │
│ • Propagate to addf (union rule)                            │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 4: Output MLIR (with sparsity info)                    │
├──────────────────────────────────────────────────────────────┤
│ func @test(%A, %B) {                                         │
│   %C = linalg.matmul {n=2, m=4,                             │
│         spa_rowmask=[T,T,F,F]} ins(%A, %B)                  │
│   %D = arith.addf {spa_rowmask=[T,T,F,F]} %C, %C           │
│   return %D                                                   │
│ }                                                             │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ Future: Optimization Passes     │
        │ • Skip zero row computation     │
        │ • Select sparse kernels         │
        │ • Generate optimized code       │
        └─────────────────────────────────┘
```

## Legend
```
▓▓▓▓  = Non-zero data (rows/values that may be non-zero)
····  = Zero data (provably zero rows/values)
T     = true (in boolean arrays)
F     = false (in boolean arrays)
```

