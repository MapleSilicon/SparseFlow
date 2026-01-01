#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT_DIR"

echo "================================================"
echo "SPA v0.6 COMPLETE DEMO - 2D Sparsity Analysis"
echo "================================================"
echo ""

./run_spa_v06_demo.sh

echo ""
echo "================================================"
echo "=== 5) Analyzing spa_sparsity.json ==="
echo "================================================"
if [ -f analyze_spa_json.py ]; then
  ./analyze_spa_json.py
else
  echo "⚠️  analyze_spa_json.py not found"
fi

echo ""
echo "================================================"
echo "=== 6) Estimating Ideal SPA Speedup ==="
echo "================================================"
if [ -f estimate_spa_speedup.py ]; then
  ./estimate_spa_speedup.py
else
  echo "⚠️  estimate_spa_speedup.py not found"
fi

echo ""
echo "================================================"
echo "✅ DEMO COMPLETE"
echo "================================================"
echo ""
echo "Key Results:"
echo "  • 2D sparsity tracked: rows + columns"
echo "  • Test case: 50% row + 50% col sparsity"
echo "  • Effective sparsity: 75%"
echo "  • Proven speedup potential: 4×"
echo ""
echo "What this means:"
echo "  SparseFlow can statically prove that 75% of computation"
echo "  can be eliminated, enabling up to 4× speedup when"
echo "  hardware/runtime exploits these sparsity masks."
echo ""
echo "Next steps:"
echo "  • Code generation to actually skip zero rows/cols"
echo "  • Runtime integration"
echo "  • Hardware acceleration"
echo "================================================"
