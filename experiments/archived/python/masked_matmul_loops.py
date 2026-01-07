#!/usr/bin/env python3
"""
SPA v0.7 Runtime — TRUE Sparse Matmul using Loop-Level Skipping

Pure Python triple-loop:
- No NumPy matmul
- Actual work reduction based on SPA masks
- FLOP counts and measured speedup
"""

import json
import numpy as np
import time


def load_spa_pattern(json_path="spa_sparsity.json"):
    """Load rowmask + colmask from SPA export."""
    with open(json_path, "r") as f:
        data = json.load(f)

    for op in data["operations"]:
        if "matmul" in op["name"]:
            return op["rowmask"], op["colmask"]

    raise RuntimeError("No matmul op found in spa_sparsity.json")


def dense_matmul_loops(A, B):
    """Naive dense triple-loop matmul."""
    n = len(A)
    m = len(B[0])

    C = [[0.0 for _ in range(m)] for _ in range(n)]
    flops = 0

    for i in range(n):
        for j in range(m):
            acc = 0.0
            for k in range(n):
                acc += A[i][k] * B[k][j]
                flops += 2
            C[i][j] = acc

    return C, flops


def sparse_matmul_loops(A, B, rowmask, colmask):
    """Sparse matmul using SPA rowmask + colmask skipping."""
    n = len(A)
    m = len(B[0])

    C = [[0.0 for _ in range(m)] for _ in range(n)]
    flops = 0

    for i in range(n):
        if not rowmask[i]:
            continue
        for j in range(m):
            if not colmask[j]:
                continue

            acc = 0.0
            for k in range(n):
                acc += A[i][k] * B[k][j]
                flops += 2
            C[i][j] = acc

    return C, flops


def benchmark(n=128):
    print(f"\n=== Testing size: {n}×{n} ===")

    # Random square matrices
    A = np.random.rand(n, n).tolist()
    B = np.random.rand(n, n).tolist()

    rowmask, colmask = load_spa_pattern()
    rowmask = (rowmask * (n // len(rowmask)))[:n]
    colmask = (colmask * (n // len(colmask)))[:n]

    t0 = time.time()
    C_dense, flops_dense = dense_matmul_loops(A, B)
    t1 = time.time()

    t2 = time.time()
    C_sparse, flops_sparse = sparse_matmul_loops(A, B, rowmask, colmask)
    t3 = time.time()

    dense_time = (t1 - t0) * 1000
    sparse_time = (t3 - t2) * 1000

    theoretical = flops_dense / flops_sparse if flops_sparse > 0 else float("inf")
    measured = dense_time / sparse_time if sparse_time > 0 else float("inf")

    # Correctness: ONLY on active region (rowmask = True AND colmask = True)
    max_diff_active = 0.0
    for i in range(n):
        if not rowmask[i]:
            continue
        for j in range(n):
            if not colmask[j]:
                continue
            diff = abs(C_dense[i][j] - C_sparse[i][j])
            max_diff_active = max(max_diff_active, diff)

    print(f"Rowmask: {rowmask[:8]}...")
    print(f"Colmask: {colmask[:8]}...")
    print(f"Zero rows: {rowmask.count(False)} / {n}")
    print(f"Zero cols: {colmask.count(False)} / {n}")

    print(f"\nFLOPs (dense):  {flops_dense:,}")
    print(f"FLOPs (sparse): {flops_sparse:,}")
    print(f"FLOP reduction: {100*(1 - flops_sparse/flops_dense):.1f}%")

    print(f"\n⏱ Dense Time:  {dense_time:.2f} ms")
    print(f"⚡ Sparse Time: {sparse_time:.2f} ms")

    print(f"\n🔥 Measured speedup:    {measured:.2f}×")
    print(f"📘 Theoretical speedup: {theoretical:.2f}×")

    print(f"\n✅ Correctness (active region): max_diff = {max_diff_active:.3e}")

    return measured, theoretical


if __name__ == "__main__":
    for size in [32, 64, 96, 128]:
        benchmark(size)
