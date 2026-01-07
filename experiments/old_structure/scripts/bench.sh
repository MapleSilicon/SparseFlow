#!/usr/bin/env bash
set -euo pipefail

echo "SparseFlow Phase-2A — Benchmark"
echo "================================"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"

cd "$BUILD_DIR"
export LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}"

# Clear caches
[[ -x "./clear_compressed_cache" ]] && ./clear_compressed_cache || true

# Run benchmarks
for test in test_1024_persistent test_2048_contenthash test_plan_sizes; do
  if [[ -x "./$test" ]]; then
    echo ""
    echo ">>> Running: $test"
    "./$test"
  fi
done

echo ""
echo "Benchmark complete ✅"
