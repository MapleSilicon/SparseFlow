#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Upgrading SparseFlow v0.1 → v0.2                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Fetch latest from GitHub
echo "Fetching latest code from GitHub..."
git fetch --all

# Check if v0.2 updates exist
echo ""
echo "Checking for v0.2 updates..."
git log HEAD..origin/v0.2-nm-sparsity --oneline | head -10

# Switch to v0.2 branch
echo ""
echo "Switching to v0.2-nm-sparsity branch..."
git checkout v0.2-nm-sparsity
git pull origin v0.2-nm-sparsity

echo ""
echo "Checking what changed..."
ls -la compiler/passes/sparseflow/SPADomain.h
ls -la runtime/src/sparseflow_runtime.cpp

# Check if NMPattern exists (v0.2 feature)
if grep -q "struct NMPattern" compiler/passes/sparseflow/SPADomain.h; then
    echo ""
    echo "✅ v0.2 code detected! NMPattern struct found."
    echo ""
    
    # Clean and rebuild
    echo "Rebuilding compiler..."
    cd compiler
    rm -rf build
    mkdir build
    cd build
    cmake \
      -DCMAKE_PREFIX_PATH=/usr/lib/llvm-19 \
      -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
      -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm \
      ..
    make -j$(nproc)
    
    cd ../..
    
    echo ""
    echo "Rebuilding runtime..."
    cd runtime
    rm -rf build
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    
    cd ../..
    
    echo ""
    echo "Testing v0.2 features..."
    
    # Test runtime
    if [ -f "runtime/build/test_nm_runtime" ]; then
        echo "Testing N:M kernels..."
        ./runtime/build/test_nm_runtime
    fi
    
    # Test compiler
    if [ -f "compiler/build/passes/SparseFlowPasses.so" ]; then
        echo ""
        echo "Testing compiler passes..."
        mlir-opt-19 \
          --load-pass-plugin=compiler/build/passes/SparseFlowPasses.so \
          --help | grep sparseflow
    fi
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              ✅ v0.2 UPGRADE COMPLETE! ✅                      ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "New features:"
    echo "  • 5 N:M patterns (1:4, 2:4, 2:8, 4:16, 8:32)"
    echo "  • Pattern-aware propagation"
    echo "  • Template-based runtime"
    echo "  • Pattern validation"
    
else
    echo ""
    echo "⚠️  v0.2 code not yet pushed to GitHub"
    echo ""
    echo "Waiting for:"
    echo "  • NMPattern implementation"
    echo "  • N:M runtime kernels"
    echo "  • Updated passes"
    echo ""
    echo "Check back soon!"
fi
