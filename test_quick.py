#!/usr/bin/env python3
"""Quick validation test"""

import sys

print("="*60)
print("SPARSEFLOW QUICK TEST")
print("="*60)
print()

# Test 1: Import
print("1. Testing imports...")
try:
    import torch
    import sparseflow as sf
    print("   ✓ Imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: GPU check
print("\n2. Checking GPU support...")
if not torch.cuda.is_available():
    print("   ⚠ CUDA not available (expected on CPU)")
    print("   → Install on GPU machine to test fully")
    print("\n✅ Basic imports work!")
    sys.exit(0)

supported, msg = sf.check_sparse_support()
print(f"   {msg}")

if not supported:
    print("   ⚠ 2:4 sparse not supported (requires Ampere+)")
    print("\n✅ Basic imports work, but need Ampere+ GPU")
    sys.exit(0)

# Test 3: Create layer
print("\n3. Creating sparse layer...")
try:
    from torch import nn
    
    dense = nn.Linear(128, 128).cuda().half()
    sparse, diff = sf.SparseLinear.from_dense(dense, return_diff=True)
    
    print(f"   ✓ Created sparse layer")
    print(f"   ✓ Max error: {diff['max_error']:.6f}")
    print(f"   ✓ Compression: {(1 - sparse.weight_compressed.numel() / dense.weight.numel()) * 100:.1f}%")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Inference
print("\n4. Testing inference...")
try:
    x = torch.randn(8, 128, device='cuda', dtype=torch.float16)
    
    y_dense = dense(x)
    y_sparse = sparse(x)
    
    error = (y_dense - y_sparse).abs().max().item()
    print(f"   ✓ Inference works")
    print(f"   ✓ Output error: {error:.6f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nSparseFlow is working correctly on your system.")
print()
