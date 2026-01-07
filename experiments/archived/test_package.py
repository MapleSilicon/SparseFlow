#!/usr/bin/env python3
"""
Test that SparseFlow package works correctly
"""
import sys
import subprocess

def test_command(cmd, description):
    print(f"\n🔧 Testing: {description}")
    print(f"   Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ Success")
        return True
    else:
        print(f"   ❌ Failed: {result.stderr}")
        return False

print("🧪 Testing SparseFlow Package Installation")
print("=" * 50)

# Test 1: Check if package is installed
test1 = test_command("python3 -c 'import sparseflow; print(f\"SparseFlow v{sparseflow.__version__}\")'", 
                    "Python import")

# Test 2: Check CLI commands exist
commands = ["sparseflow-demo", "sparseflow-analyze", "sparseflow-benchmark"]
for cmd in commands:
    test_command(f"which {cmd}", f"CLI command: {cmd}")

# Test 3: Run a simple analysis if test file exists
import os
test_mlir = "tests/test_spa_v6_full_2d.mlir"
if os.path.exists(test_mlir):
    test_command(f"sparseflow-analyze {test_mlir}", f"Analyze {test_mlir}")

print("\n" + "=" * 50)
print("✅ Package test complete!")
print("\nNext steps:")
print("1. Build compiler: cd compiler && mkdir -p build && cd build && cmake .. -DCMAKE_PREFIX_PATH=/usr/lib/llvm-19 && make")
print("2. Build runtime: cd runtime && mkdir -p build && cd build && cmake .. && make")
print("3. Run full demo: sparseflow-demo")
