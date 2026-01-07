#!/usr/bin/env python3
"""
Quick test to verify SparseFlow works
"""
import sys
sys.path.insert(0, '.')

from sparseflow.cli import analyze_mlir, benchmark_sparsity

print("🧪 Testing SparseFlow installation...")

# Test with the existing example
import os
mlir_file = "../tests/test_spa_v6_full_2d.mlir"
if os.path.exists(mlir_file):
    print(f"✓ Found test MLIR: {mlir_file}")
    analyze_mlir(mlir_file)
    
    if os.path.exists("spa_sparsity.json"):
        print("\n✓ Analysis successful, running benchmark...")
        benchmark_sparsity()
    else:
        print("\n✗ Analysis failed: spa_sparsity.json not found")
else:
    print(f"✗ Test MLIR not found: {mlir_file}")
    print("Please run from SparseFlow root directory")
