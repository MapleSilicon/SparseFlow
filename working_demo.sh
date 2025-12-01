#!/bin/bash
echo "===================================================================="
echo "🧠 SPARSEFLOW WORKING DEMO - Structured Sparsity Acceleration"
echo "===================================================================="
echo ""

# Clean and build
echo "🔧 Building SparseFlow..."
rm -rf compiler/build runtime/build
./build_all.sh > build.log 2>&1

echo ""
echo "📊 DEMO: Live Pipeline Demonstration"
echo "--------------------------------------"

# Test with 32x32
echo "Testing 32x32 matrix:"
cd compiler/build
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  -pass-pipeline='builtin.module(sparseflow-export-metadata)' \
  ../../test/sparseflow-32x32.mlir > /dev/null

echo "Generated JSON:"
cat hardware_config.json

echo ""
echo "Runtime output:"
cd ../../runtime/build
./sparseflow_test

echo ""
echo "✅ Demo completed successfully!"
