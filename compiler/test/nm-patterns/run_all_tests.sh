#!/bin/bash

echo "═══════════════════════════════════════════════════════════"
echo "   Running N:M Pattern Tests"
echo "═══════════════════════════════════════════════════════════"
echo ""

PASS=0
FAIL=0

for test in test_*.mlir; do
    echo "Testing: $test"
    
    mlir-opt-19 \
        --allow-unregistered-dialect \
        --load-pass-plugin=../../build/passes/SparseFlowPasses.so \
        --pass-pipeline="builtin.module(sparseflow-spa, func.func(sparseflow-rewrite-matmul))" \
        "$test" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✓ PASS"
        ((PASS++))
    else
        echo "  ✗ FAIL"
        ((FAIL++))
    fi
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "Results: $PASS passed, $FAIL failed"
echo "═══════════════════════════════════════════════════════════"

exit $FAIL
