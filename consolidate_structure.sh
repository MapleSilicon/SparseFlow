#!/bin/bash
set -e

echo "=========================================="
echo "Consolidating SparseFlow Structure"
echo "=========================================="
echo ""
echo "This will merge old directories into new structure"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Ensure new structure exists
mkdir -p src/sparseflow
mkdir -p kernels/cuda
mkdir -p experiments/old_structure

echo "Moving old compiler/ directory..."
[ -d "compiler" ] && git mv compiler experiments/old_structure/

echo "Moving old mlir/ directory..."
[ -d "mlir" ] && git mv mlir experiments/old_structure/

echo "Moving old runtime/ directory..."
[ -d "runtime" ] && git mv runtime experiments/old_structure/

echo "Moving old gpu/ directory..."
[ -d "gpu" ] && git mv gpu experiments/old_structure/

echo "Moving old python/ directory..."
[ -d "python" ] && git mv python experiments/old_structure/

echo "Moving demos and examples..."
[ -d "demos" ] && git mv demos experiments/old_structure/
[ -d "examples" ] && git mv examples experiments/old_structure/

echo "Moving random directories..."
[ -d "SparseFlow" ] && git mv SparseFlow experiments/old_structure/
[ -d "sparseflow" ] && git mv sparseflow experiments/old_structure/
[ -d "sparseflow_package" ] && git mv sparseflow_package experiments/old_structure/
[ -d "sparseflow_pipeline" ] && git mv sparseflow_pipeline experiments/old_structure/
[ -d "week3" ] && git mv week3 experiments/old_structure/
[ -d "github_upload" ] && git mv github_upload experiments/old_structure/
[ -d "scripts" ] && git mv scripts experiments/old_structure/

echo "Moving old include/ headers..."
[ -d "include" ] && git mv include experiments/old_structure/

echo "Cleaning up files..."
[ -f ".github_release_v0.3-alpha.md" ] && git mv .github_release_v0.3-alpha.md experiments/old_structure/
[ -f "README.md.backup" ] && git mv README.md.backup experiments/old_structure/
[ -f "setup.py" ] && git mv setup.py experiments/old_structure/

echo ""
echo "✅ Consolidation complete!"
echo ""
echo "Now run: ls -1"
