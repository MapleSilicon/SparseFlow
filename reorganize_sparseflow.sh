#!/bin/bash
set -e

echo "=========================================="
echo "SparseFlow Repository Reorganization"
echo "=========================================="
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Creating directory structure..."
mkdir -p docs src/sparseflow/SparsityPropagation src/sparseflow/SparseGemmLowering
mkdir -p kernels/cuda kernels/ptx benchmarks/results
mkdir -p tests/unit tests/integration experiments/archived tools

echo "Moving production passes..."
[ -f "SparsityPropagation.cpp" ] && git mv SparsityPropagation.cpp src/sparseflow/SparsityPropagation/ || true
[ -f "SparseGemmLowering.cpp" ] && git mv SparseGemmLowering.cpp src/sparseflow/SparseGemmLowering/ || true

echo "Moving kernel implementations..."
for file in kernel_*.cu; do [ -f "$file" ] && git mv "$file" kernels/cuda/ || true; done
for file in kernel_*.ptx; do [ -f "$file" ] && git mv "$file" kernels/ptx/ || true; done
for file in *.cu; do [ -f "$file" ] && git mv "$file" kernels/cuda/ || true; done
for file in *.ptx; do [ -f "$file" ] && git mv "$file" kernels/ptx/ || true; done

echo "Moving benchmarks..."
for file in benchmark*.py run_bench*.py *benchmark*.py; do [ -f "$file" ] && git mv "$file" benchmarks/ || true; done
[ -f "kernel_selection_cache.db" ] && git mv kernel_selection_cache.db benchmarks/ || true

echo "Moving test MLIR files..."
for file in test_*.mlir *_test.mlir; do [ -f "$file" ] && git mv "$file" tests/unit/ || true; done
for file in matmul*.mlir gemm*.mlir sparse*.mlir; do [ -f "$file" ] && git mv "$file" tests/integration/ || true; done

echo "Moving tools..."
for file in kernel_cache*.py visualize*.py plot*.py generate*.py; do [ -f "$file" ] && git mv "$file" tools/ || true; done

echo "Archiving experiments..."
for file in debug_* temp_* scratch_* experiment_* old_* backup_* *.txt *.log *.out; do
    [ -f "$file" ] && [[ ! "$file" =~ ^(README|LICENSE|CMakeLists) ]] && git mv "$file" experiments/archived/ || true
done

echo "Moving test scripts..."
for file in test*.py check*.py verify*.py; do
    [ -f "$file" ] && [[ ! "$file" =~ (test_suite|test_runner) ]] && git mv "$file" experiments/archived/ || true
done

echo "Moving documentation..."
for file in *.md; do [ -f "$file" ] && [ "$file" != "README.md" ] && git mv "$file" docs/ || true; done

touch benchmarks/results/.gitkeep experiments/archived/.gitkeep

cat > .gitignore << 'EOF'
build/
*.o
*.so
*.a
__pycache__/
*.pyc
benchmarks/results/*.json
benchmarks/results/*.csv
benchmarks/results/*.png
!benchmarks/results/.gitkeep
*.db
*.db-journal
experiments/archived/*.txt
experiments/archived/*.log
.vscode/
.DS_Store
*.tmp
EOF

git add .gitignore benchmarks/results/.gitkeep experiments/archived/.gitkeep

echo ""
echo "✅ Reorganization complete!"
echo ""
echo "Next: python3 update_imports.py"
