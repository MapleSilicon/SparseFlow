#!/usr/bin/env bash
set -euo pipefail

echo "===================================="
echo "SparseFlow Phase-3 — Week-1 Bench"
echo "Dense vs Dense-Fused"
echo "===================================="
echo ""

ROOT_BIN="/root/test_phase3_week1"
ARTDIR="artifacts"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ARTDIR}/phase3_week1_${STAMP}.txt"

mkdir -p "${ARTDIR}"

# Capture environment
{
  echo "== Environment =="
  echo "Date: $(date -u) (UTC)"
  echo "Host: $(hostname)"
  echo ""
  echo "== nvidia-smi =="
  (nvidia-smi || true)
  echo ""
  echo "== nvcc =="
  (nvcc --version || true)
  echo ""
} | tee "${OUT}"

# Build if missing
if [[ ! -x "${ROOT_BIN}" ]]; then
  echo ""
  echo "Binary not found at ${ROOT_BIN}. Building..."
  pushd /root >/dev/null
  rm -rf CMakeCache.txt CMakeFiles
  cmake ~/CMakeLists.txt
  make test_phase3_week1 -j"$(nproc)"
  popd >/dev/null
fi

echo "" | tee -a "${OUT}"
echo "== Results ==" | tee -a "${OUT}"
echo "" | tee -a "${OUT}"

# Run and capture
"${ROOT_BIN}" | tee -a "${OUT}"

echo ""
echo "Saved: ${OUT}"
