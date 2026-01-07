#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/runtime"

echo "=== Configuring ==="
cmake -B build -S .

echo "=== Building ==="
cmake --build build -j4

echo "=== Running Benchmark ==="
./build/benchmark_sparse
