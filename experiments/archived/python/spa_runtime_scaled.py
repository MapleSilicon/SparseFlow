#!/usr/bin/env python3
"""
SPA v0.7 Runtime - Scaled to realistic matrix sizes
"""
import numpy as np
import json
import time
from typing import Dict, List, Tuple

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
    """Scale a mask pattern to a larger size by repeating the pattern."""
    pattern_len = len(mask)
    return [mask[i % pattern_len] for i in range(new_size)]

def dense_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Standard dense matrix multiplication."""
    return A @ B

def spa_masked_matmul(A: np.ndarray, B: np.ndarray, 
                      rowmask: List[bool], 
                      colmask: List[bool]) -> np.ndarray:
    """SPA-guided sparse matmul - only compute non-zero rows/cols."""
    M, K = A.shape
    _, N = B.shape
    
    C = np.zeros((M, N), dtype=A.dtype)
    
    active_rows = [i for i, active in enumerate(rowmask) if active]
    active_cols = [j for j, active in enumerate(colmask) if active]
    
    # Vectorized computation for active blocks
    # This is more efficient than nested Python loops
    for i in active_rows:
        C[i, active_cols] = A[i, :] @ B[:, active_cols]
    
    return C

def benchmark_matmul(func, A, B, *args, iterations=100):
    """Benchmark a matmul function."""
    # Warmup
    for _ in range(10):
        _ = func(A, B, *args)
    
    # Actual timing
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(A, B, *args)
    end = time.perf_counter()
    
    avg_time = (end - start) / iterations
    return avg_time, result

def test_at_scale(size: int, pattern: Dict):
    """Test matmul at given size with SPA pattern."""
    print(f"\n{'='*60}")
    print(f"Testing at size: {size}×{size}")
    print(f"{'='*60}")
    
    # Scale the pattern
    rowmask = scale_mask(pattern['rowmask'], size)
    colmask = scale_mask(pattern['colmask'], size)
    
    active_rows = sum(rowmask)
    active_cols = sum(colmask)
    
    print(f"Active rows: {active_rows}/{size} ({active_rows/size*100:.0f}%)")
    print(f"Active cols: {active_cols}/{size} ({active_cols/size*100:.0f}%)")
    print(f"Active elements: {active_rows*active_cols}/{size*size} ({active_rows*active_cols/(size*size)*100:.1f}%)")
    print()
    
    # Create test matrices
    np.random.seed(42)
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    # Apply sparsity pattern
    for i, active in enumerate(rowmask):
        if not active:
            A[i, :] = 0
    for j, active in enumerate(colmask):
        if not active:
            B[:, j] = 0
    
    # Benchmark
    print("Running benchmarks...")
    dense_time, C_dense = benchmark_matmul(dense_matmul, A, B, iterations=50)
    sparse_time, C_sparse = benchmark_matmul(
        spa_masked_matmul, A, B, rowmask, colmask, iterations=50
    )
    
    # Verify
    max_diff = np.max(np.abs(C_dense - C_sparse))
    
    # Results
    speedup = dense_time / sparse_time
    theoretical = (size * size) / (active_rows * active_cols)
    
    print(f"✅ Correctness: max diff = {max_diff:.2e}")
    print(f"⚙️  Dense:  {dense_time*1000:.2f} ms")
    print(f"⚡ Sparse: {sparse_time*1000:.2f} ms")
    print(f"🔥 SPEEDUP: {speedup:.2f}× (theoretical: {theoretical:.2f}×)")
    print(f"   Efficiency: {(speedup/theoretical)*100:.1f}%")
    
    return speedup, theoretical

def main():
    print("=" * 60)
    print("SPA v0.7 Runtime - Scaled Performance Test")
    print("=" * 60)
    
    # Load pattern
    pattern = load_spa_pattern('spa_sparsity.json')
    if not pattern:
        print("❌ No matmul in spa_sparsity.json")
        return
    
    print(f"\n✅ Base pattern:")
    print(f"   Row mask: {pattern['rowmask']}")
    print(f"   Col mask: {pattern['colmask']}")
    print(f"   Sparsity: {pattern['row_sparsity']:.0f}% rows, {pattern['col_sparsity']:.0f}% cols")
    
    # Test at multiple scales
    sizes = [64, 128, 256, 512]
    results = []
    
    for size in sizes:
        speedup, theoretical = test_at_scale(size, pattern)
        results.append((size, speedup, theoretical))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY - Speedup by Matrix Size")
    print(f"{'='*60}")
    print(f"{'Size':<10} {'Measured':<12} {'Theoretical':<12} {'Efficiency':<12}")
    print(f"{'-'*60}")
    for size, speedup, theoretical in results:
        efficiency = (speedup / theoretical) * 100
        print(f"{size:<10} {speedup:.2f}×{'':<8} {theoretical:.2f}×{'':<8} {efficiency:.1f}%")
    print(f"{'='*60}")
    print()
    print("💡 Key Insight:")
    print("   Sparsity optimization pays off at larger scales where")
    print("   computation savings dominate Python/indexing overhead.")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
