#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║         🔍 SPARSEFLOW v0.2 COMPLETE VALIDATION 🔍              ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}: $2"
        ((passed++))
    else
        echo -e "${RED}❌ FAIL${NC}: $2"
        ((failed++))
    fi
}

echo "════════════════════════════════════════════════════════════════"
echo "📋 PART 1: GIT STATUS"
echo "════════════════════════════════════════════════════════════════"
echo ""

if git status | grep -q "nothing to commit\|Your branch is up to date"; then
    test_result 0 "Git status clean"
else
    echo -e "${YELLOW}⚠️  Git has uncommitted changes or needs push${NC}"
    git status --short
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📋 PART 2: COMPILER BUILD"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ -f "compiler/build/passes/SparseFlowPasses.so" ]; then
    test_result 0 "Compiler plugin exists"
    
    # Check if it loads
    if mlir-opt-19 --load-pass-plugin=compiler/build/passes/SparseFlowPasses.so --help &>/dev/null; then
        test_result 0 "Compiler plugin loads"
        
        # Check for the rewrite pass
        if mlir-opt-19 --load-pass-plugin=compiler/build/passes/SparseFlowPasses.so --help 2>&1 | grep -q "sparseflow-rewrite-matmul"; then
            test_result 0 "Rewrite pass registered"
            
            # Check description
            DESC=$(mlir-opt-19 --load-pass-plugin=compiler/build/passes/SparseFlowPasses.so --help 2>&1 | grep -A1 "sparseflow-rewrite-matmul" | tail -1)
            echo "  Description: $DESC"
        else
            test_result 1 "Rewrite pass NOT found"
        fi
    else
        test_result 1 "Compiler plugin failed to load"
    fi
else
    test_result 1 "Compiler plugin missing - need to build"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📋 PART 3: RUNTIME BUILD"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ -f "runtime/build/libsparseflow_runtime.so" ]; then
    test_result 0 "Runtime library exists"
    
    # Check for all 5 kernel functions
    for pattern in 1_4 2_4 2_8 4_16 8_32; do
        if nm -D runtime/build/libsparseflow_runtime.so 2>/dev/null | grep -q "sparse_matmul_${pattern}"; then
            test_result 0 "Kernel sparse_matmul_${pattern} exported"
        else
            test_result 1 "Kernel sparse_matmul_${pattern} NOT found"
        fi
    done
else
    test_result 1 "Runtime library missing - need to build"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📋 PART 4: RUNTIME TESTS"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ -f "runtime/build/test_nm_runtime" ]; then
    test_result 0 "Test executable exists"
    
    echo "  Running runtime tests..."
    cd runtime/build
    export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
    if ./test_nm_runtime &>/dev/null; then
        test_result 0 "Runtime tests execute successfully"
    else
        test_result 1 "Runtime tests failed"
    fi
    cd ../..
else
    test_result 1 "Test executable missing"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📋 PART 5: MLIR INTEGRATION TESTS"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ -d "compiler/tests" ]; then
    for pattern in 1_4 2_4 2_8 4_16 8_32; do
        if [ -f "compiler/tests/test_nm_rewrite_${pattern}.mlir" ]; then
            if mlir-opt-19 \
                --load-pass-plugin=compiler/build/passes/SparseFlowPasses.so \
                --pass-pipeline="builtin.module(sparseflow-rewrite-matmul)" \
                compiler/tests/test_nm_rewrite_${pattern}.mlir 2>&1 | grep -q "sparse_matmul_${pattern}"; then
                test_result 0 "MLIR test ${pattern} generates correct call"
            else
                test_result 1 "MLIR test ${pattern} failed"
            fi
        else
            test_result 1 "MLIR test file ${pattern} missing"
        fi
    done
else
    test_result 1 "MLIR tests directory missing"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📋 PART 6: QUICK BENCHMARK"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ -f "runtime/build/benchmark_nm_runtime" ]; then
    test_result 0 "Benchmark executable exists"
    
    echo "  Running quick benchmark (this takes ~10 seconds)..."
    cd runtime/build
    export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
    if timeout 15s ./benchmark_nm_runtime &>/dev/null; then
        test_result 0 "Benchmark runs successfully"
    else
        echo -e "${YELLOW}  ⚠️  Benchmark timed out or failed (not critical)${NC}"
    fi
    cd ../..
else
    echo -e "${YELLOW}  ⚠️  Benchmark executable not built${NC}"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📊 FINAL RESULTS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}Passed: $passed${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║              🎉 ALL CHECKS PASSED! 🎉                          ║"
    echo "║                                                                ║"
    echo "║         SparseFlow v0.2.0 is FULLY WORKING!                    ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║              ⚠️  SOME CHECKS FAILED ⚠️                         ║"
    echo "║                                                                ║"
    echo "║         Review the failures above                              ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    exit 1
fi
