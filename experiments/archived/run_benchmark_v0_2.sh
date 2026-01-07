#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Go to repo root (in case script is called from somewhere else)
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       SparseFlow v0.2 N:M Pattern Benchmark Re-Run            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Make sure runtime build dir exists
if [ ! -d "runtime/build" ]; then
  echo "➜ Creating runtime/build directory..."
  mkdir -p runtime/build
fi

cd runtime/build

echo "➜ Configuring runtime (cmake ..)…"
cmake .. >/dev/null

echo "➜ Building benchmark_nm_runtime target…"
if make -j8 benchmark_nm_runtime; then
  echo -e "${GREEN}✅ Build success${NC}"
else
  echo -e "${RED}❌ Build failed${NC}"
  exit 1
fi

if [ ! -f "./benchmark_nm_runtime" ]; then
  echo -e "${RED}❌ benchmark_nm_runtime executable not found${NC}"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_LOCAL="BENCHMARK_RUN_v0.2_${TS}.txt"
OUT_ROOT="$ROOT_DIR/BENCHMARK_RUN_v0.2_${TS}.txt"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🏃 Running full N:M benchmark suite (this may take ~30–60s)…"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Run and tee both in build dir and repo root
./benchmark_nm_runtime | tee "$OUT_LOCAL" | tee "$OUT_ROOT" >/dev/null

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📁 Benchmark saved to:"
echo "  • $OUT_LOCAL"
echo "  • $OUT_ROOT"
echo "════════════════════════════════════════════════════════════════"
