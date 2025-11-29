#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/src/SparseFlow/runtime/build"

echo "=== Running SparseFlow v0.1 Runtime Test ==="
./sparseflow_test
