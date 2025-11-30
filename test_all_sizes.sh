#!/bin/bash
echo "=== SparseFlow Full Test Suite ==="
echo ""

echo "=== Testing 64x64 (Default) ==="
./build_all.sh

echo ""
echo "=== Testing 32x32 ==="
SPARSEFLOW_MLIR_FILE=compiler/test/sparseflow-32x32.mlir ./build_all.sh

echo ""
echo "=== JSON Verification ==="
echo "64x64 JSON:"
cat compiler/build/hardware_config.json
echo ""
echo "32x32 JSON:" 
SPARSEFLOW_MLIR_FILE=compiler/test/sparseflow-32x32.mlir ./build_all.sh > /dev/null 2>&1
cat compiler/build/hardware_config.json

echo ""
echo "=== Test Complete ==="

