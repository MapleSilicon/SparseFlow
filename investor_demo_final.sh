#!/bin/bash
set -e

echo "===================================================================="
echo "🧠 SPARSEFLOW INVESTOR DEMO - Production Ready v0.1"
echo "===================================================================="
echo ""

echo "🔧 Building full pipeline (compiler + runtime)..."
./build_all.sh > /tmp/sparseflow_investor_build.log 2>&1

echo ""
echo "📊 DEMO 1: Compiler Pipeline (MLIR → JSON)"
echo "------------------------------------------"
cd compiler/build

echo "Step 1: Running sparseflow-export-metadata on 32×32 matmul..."
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  -pass-pipeline="builtin.module(func.func(sparseflow-export-metadata))" \
  ../test/sparseflow-32x32.mlir 2>&1 | grep -E "Exporting matmul|totalMACs|executedMACs" || true

echo ""
echo "Step 2: Generated hardware_config.json:"
if [ -f hardware_config.json ]; then
  python3 -m json.tool hardware_config.json
else
  echo "❌ ERROR: hardware_config.json not found"
  exit 1
fi

echo ""
echo "📈 DEMO 2: Runtime Execution (JSON → Hardware Model)"
echo "----------------------------------------------------"
cd ../../runtime/build
./sparseflow_test 2>&1 | grep -E "Loading hardware configuration|Programming MapleSilicon Hardware|Dimensions|Total MACs|Executed MACs|Compute Efficiency|Theoretical Speedup" || true

echo ""
echo "🎯 DEMO COMPLETE"
echo "  - MLIR → JSON export via sparseflow-export-metadata"
echo "  - JSON-driven runtime configuration"
echo "  - 2:4 sparsity with ~50% MAC reduction and ~2.0x theoretical speedup"
