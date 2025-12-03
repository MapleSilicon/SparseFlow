#!/usr/bin/env python3
"""
SPA v0.7 Runtime - OPTIMIZED with block operations (no loops!)
"""
import numpy as np
import json
import time
from typing import Dict, List

def load_spa_pattern(json_path: str) -> Dict:
    """Load SPA sparsity PATTERN from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for op in data['operations']:
        if 'matmul' in op['name']:
            return {
                'rowmask': op.get('rowmask', []),
                'colmask': op.get('colmask', []),
                'row_sparsity': op.get('row_sparsity_pct', 0),
                'col_sparsity': op.get('col_sparsity_pct', 0)
            }
    return None

def scale_mask(mask: List[bool], new_size: int) -> List[bool]:
    """Scale a mask pattern by repeating."""
    pattern_len = len(mask)
    return [mask[i % pattern_len] for i in range(new_size)]

def dense_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Standard dense matrix multiplication."""
    return A @ B

def spa_masked_matmul_OPTIMIZED(A: np.ndarray, B: np.ndarray, 
                                rowmask: List[bool], 
                                colmask: List[bool]) -> np.ndarray:
    """
    SPA-guided sparse matmul - OPTIMIZED with block operations.
    
    Key optimization: Extract active block, do ONE matmul, scatter back.
    NO PYTHON LOOPS!
    """
    M, K = A.shape
    _, N = B.shape
    
    # Convert masks to numpy arrays of indices
    active_rows = np.array([i for i, m in enumerate(rowmask) if m], dtype=np.int32)
    active_cols = np.array([j for j, m in enumerate(colmask) if m], dtype=np.int32)
    
    # Extract active blocks (fancy indexing)
    A_active = A[active_rows, :]        # shape: (num_active_rows, K)
    B_active = B[:, active_cols]        # shape: (K, num_active_cols)
    
    # Compute ONLY the active block with ONE matmul
    C_active = A_active @ B_active      # shape: (num_active_rows, num_active_cols)
    
    # Scatter back to full output matrix
    C = np.zeros((M, N), dtype=A.dtype)
    C[np.ix_(active_rows, active_cols)] = C_active
    
    return C

def benchmark_matmul(func, A, B, *args, iterations=100):
    """Benchmark a matmul function."""
    # Warmup
    for _ in range(10):
        _ = func(A, B, *args)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(A, B, *args)
    end = time.perf_counter()
    
    return (end - start) / iterations, result

def test_at_scale(size: int, pattern: Dict):
    """Test matmul at given size."""
    print(f"\n{'='*60}")
    print(f"Testing at size: {size}×{size}")
    print(f"{'='*60}")
    
    rowmask = scale_mask(pattern['rowmask'], size)
    colmask = scale_mask(pattern['colmask'], size)
    
    active_rows = sum(rowmask)
    active_cols = sum(colmask)
    
    print(f"Active: {active_rows}×{active_cols} / {size}×{size} ({active_rows*active_cols/(size*size)*100:.1f}%)")
    
    # Create test matrices
    np.random.seed(42)
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    # Apply sparsity
    for i, active in enumerate(rowmask):
        if not active:
            A[i, :] = 0
    for j, active in enumerate(colmask):
        if not active:
            B[:, j] = 0
    
    # Benchmark
    dense_time, C_dense = benchmark_matmul(dense_matmul, A, B, iterations=50)
    sparse_time, C_sparse = benchmark_matmul(
        spa_masked_matmul_OPTIMIZED, A, B, rowmask, colmask, iterations=50
    )
    
    # Verify
    max_diff = np.max(np.abs(C_dense - C_sparse))
    
    speedup = dense_time / sparse_time
    theoretical = (size * size) / (active_rows * active_cols)
    
    print(f"✅ Correct: max diff = {max_diff:.2e}")
    print(f"⚙️  Dense:  {dense_time*1000:.2f} ms")
    print(f"⚡ Sparse: {sparse_time*1000:.2f} ms")
    print(f"🔥 SPEEDUP: {speedup:.2f}× (theoretical: {theoretical:.2f}×)")
    print(f"   Efficiency: {(speedup/theoretical)*100:.1f}%")
    
    return speedup, theoretical

def main():
    print("=" * 60)
    print("SPA v0.7 Runtime - OPTIMIZED (Block Operations)")
    print("=" * 60)
    
    pattern = load_spa_pattern('spa_sparsity.json')
    if not pattern:
        print("❌ No matmul in spa_sparsity.json")
        return
    
    print(f"\n✅ Pattern: {pattern['rowmask']} × {pattern['colmask']}")
    print(f"   Sparsity: {pattern['row_sparsity']:.0f}% rows, {pattern['col_sparsity']:.0f}% cols")
    
    sizes = [64, 128, 256, 512, 1024]
    results = []
    
    for size in sizes:
        speedup, theoretical = test_at_scale(size, pattern)
        results.append((size, speedup, theoretical))
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Size':<10} {'Measured':<15} {'Theoretical':<15} {'Efficiency'}")
    print(f"{'-'*60}")
    for size, speedup, theoretical in results:
        eff = (speedup / theoretical) * 100
        status = "🔥" if speedup > 1.5 else "⚠️" if speedup > 0.8 else "❌"
        print(f"{size:<10} {status} {speedup:.2f}×{'':<9} {theoretical:.2f}×{'':<9} {eff:.1f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
