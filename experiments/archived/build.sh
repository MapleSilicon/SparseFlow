#!/bin/bash
set -e

echo "🔨 Building SparseFlow v3.0 (Epilogue Fusion)"

# Create build directory
mkdir -p build
cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure

echo "✅ Build complete!"
echo "Library: build/libsparseflow.so"
