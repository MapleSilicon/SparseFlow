#!/usr/bin/env bash
set -euo pipefail

echo "SparseFlow Phase-2A — Environment Check"
echo "========================================"

for bin in git cmake g++ nvcc; do
  command -v "$bin" >/dev/null 2>&1 || { echo "ERROR: missing $bin"; exit 1; }
done

echo "[OK] nvcc: $(nvcc --version | tail -n1)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[OK] GPU available"
  nvidia-smi -L || true
else
  echo "[WARN] nvidia-smi not found (build OK, GPU tests require runtime)"
fi

echo "Environment OK ✅"
