#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "SparseFlow Bootstrap"
echo "========================================="
echo ""

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc not found"
    echo "   Install CUDA Toolkit or add to PATH"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "✅ CUDA: $CUDA_VERSION"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found"
    echo "   Build will work, GPU tests require NVIDIA driver"
else
    echo "✅ GPU available"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
fi

# Check cuSPARSELt
if [ ! -f /usr/local/cuda/lib64/libcusparseLt.so ]; then
    echo "❌ ERROR: cuSPARSELt not found"
    echo "   Requires CUDA 11.x+ with cuSPARSELt"
    exit 1
fi
echo "✅ cuSPARSELt found"

# Check SQLite3
if ! command -v sqlite3 &> /dev/null; then
    echo "❌ ERROR: sqlite3 not found"
    echo "   Install: apt-get install sqlite3 libsqlite3-dev"
    exit 1
fi
echo "✅ SQLite3 available"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ ERROR: cmake not found"
    exit 1
fi
echo "✅ CMake available"

echo ""
echo "All dependencies satisfied ✅"
echo ""
echo "Building SparseFlow..."
./scripts/build.sh

echo ""
echo "========================================="
echo "✅ BOOTSTRAP COMPLETE"
echo "========================================="
echo ""
echo "Next step: ./scripts/validate.sh"
