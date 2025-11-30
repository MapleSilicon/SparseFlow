#!/bin/bash
echo "=== SparseFlow Test Suite ==="
echo ""

for size in 32x32 64x64 128x128 1024x1024; do
  if [ -f "compiler/test/sparseflow-$size.mlir" ]; then
    echo "=== Testing $size matmul ==="
    SPARSEFLOW_MLIR_FILE=compiler/test/sparseflow-$size.mlir ./build_all.sh 2>&1 | grep -E "Total MACs|Executed MACs|Speedup"
    echo ""
  fi
done

echo "=== All tests complete ==="
