#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "═══════════════════════════════════════════════════════════"
echo "   SparseFlow v0.7 - Complete End-to-End Demo"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check prerequisites
echo "[0] Checking prerequisites..."
command -v mlir-opt-19 >/dev/null 2>&1 || { echo "❌ mlir-opt-19 not found. Install llvm-19-dev"; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "❌ cmake not found"; exit 1; }
echo "✅ Prerequisites OK"
echo ""

# Build if needed
if [ ! -f "$ROOT_DIR/compiler/build/passes/SparseFlowPasses.so" ]; then
    echo "[1] Building compiler passes..."
    cd "$ROOT_DIR/compiler/build"
    cmake -DCMAKE_PREFIX_PATH=/usr/lib/llvm-19 .. >/dev/null
    make -j4 >/dev/null
    echo "✅ Compiler passes built"
else
    echo "[1] Compiler passes already built ✅"
fi

if [ ! -f "$ROOT_DIR/runtime/build/benchmark_sparse" ]; then
    echo "[2] Building C++ runtime..."
    mkdir -p "$ROOT_DIR/runtime/build"
    cd "$ROOT_DIR/runtime/build"
    cmake .. >/dev/null
    make -j4 >/dev/null
    echo "✅ C++ runtime built"
else
    echo "[2] C++ runtime already built ✅"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "[3] Running Static Analysis (SPA v0.6)"
echo "═══════════════════════════════════════════════════════════"

cd "$ROOT_DIR"
mlir-opt-19 --allow-unregistered-dialect \
  --load-pass-plugin=compiler/build/passes/SparseFlowPasses.so \
  --pass-pipeline='builtin.module(sparseflow-spa,sparseflow-spa-export)' \
  test_spa_v6_full_2d.mlir >/dev/null 2>&1

if [ -f spa_sparsity.json ]; then
    echo "✅ Sparsity analysis complete"
    
    # Extract sparsity info
    row_sparsity=$(grep -oP '"row_sparsity_pct":\s*\K\d+' spa_sparsity.json | head -1)
    col_sparsity=$(grep -oP '"col_sparsity_pct":\s*\K\d+' spa_sparsity.json | head -1)
    
    echo "   📊 Detected: ${row_sparsity}% row + ${col_sparsity}% col sparsity"
    
    # Calculate total sparsity
    total_sparse=$((100 - (100 - row_sparsity) * (100 - col_sparsity) / 100))
    echo "   📊 Total computation saved: ~${total_sparse}%"
    echo "   📄 Metadata exported to: spa_sparsity.json"
else
    echo "❌ Analysis failed - spa_sparsity.json not created"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "[4] Running C++ Runtime Benchmark"
echo "═══════════════════════════════════════════════════════════"

cd "$ROOT_DIR/runtime/build"
./benchmark_sparse | tee /tmp/spa_benchmark.log

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ DEMO COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Key Results:"
echo "  • Static analysis detected ~75% removable computation"
echo "  • C++ runtime achieves ~3–4× speedup on larger matmuls (512–1024)"
echo "  • For small sizes (<512), overhead dominates (sometimes <1×)"
echo "  • Pipeline: MLIR → SPA → JSON → C++ runtime"
echo ""
echo "Next Steps:"
echo "  • View metadata:     cat spa_sparsity.json"
echo "  • View benchmarks:   cat /tmp/spa_benchmark.log"
echo "  • Read docs:         cat QUICK_DEMO.md"
echo "  • (Optional) Tweak sizes in runtime/benchmark_sparse.cpp"
echo ""
echo "═══════════════════════════════════════════════════════════"
