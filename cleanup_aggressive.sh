#!/bin/bash
set -e

echo "=========================================="
echo "AGGRESSIVE Repository Cleanup"
echo "=========================================="
echo ""
echo "This will move ALL loose files to organized locations."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Ensure directories exist
mkdir -p experiments/archived/{mlir,python,outputs,configs}
mkdir -p tests/integration
mkdir -p kernels/cuda
mkdir -p benchmarks

echo "Moving ALL .mlir files to tests..."
for file in *.mlir; do
    [ -f "$file" ] && git mv "$file" tests/integration/ && echo "  → $file"
done

echo "Moving ALL .cu files to kernels..."
for file in *.cu; do
    [ -f "$file" ] && git mv "$file" kernels/cuda/ && echo "  → $file"
done

echo "Moving ALL Python scripts (except setup/README)..."
for file in *.py; do
    if [ -f "$file" ]; then
        # Keep only essential root scripts
        if [[ ! "$file" =~ ^(setup|__init__|reorganize|cleanup|update_imports|verify) ]]; then
            git mv "$file" experiments/archived/python/ && echo "  → $file"
        fi
    fi
done

echo "Moving ALL .txt/.log/.out files..."
for file in *.{txt,log,out}; do
    [ -f "$file" ] && git mv "$file" experiments/archived/outputs/ && echo "  → $file"
done

echo "Moving ALL .json files..."
for file in *.json; do
    [ -f "$file" ] && [[ ! "$file" =~ (package|tsconfig|config) ]] && git mv "$file" experiments/archived/configs/ && echo "  → $file"
done

echo "Moving ALL .sh scripts (except this one)..."
for file in *.sh; do
    [ -f "$file" ] && [[ "$file" != "cleanup_aggressive.sh" ]] && [[ "$file" != "reorganize_sparseflow.sh" ]] && git mv "$file" experiments/archived/ && echo "  → $file"
done

echo "Moving remaining loose files..."
for file in *; do
    if [ -f "$file" ]; then
        # Keep only essential files in root
        if [[ ! "$file" =~ ^(README|LICENSE|CMakeLists|\.git|Makefile|setup\.|\.clang|cleanup_aggressive\.sh|reorganize_sparseflow\.sh|update_imports\.py|verify_reorganization\.py) ]]; then
            # Check file extension and move appropriately
            ext="${file##*.}"
            case "$ext" in
                cpp|h|hpp)
                    git mv "$file" experiments/archived/ && echo "  → $file (loose C++ file)"
                    ;;
                *)
                    git mv "$file" experiments/archived/ && echo "  → $file"
                    ;;
            esac
        fi
    fi
done

echo ""
echo "✅ Aggressive cleanup complete!"
echo ""
echo "Root directory should now only have:"
echo "  - README.md"
echo "  - LICENSE (if exists)"
echo "  - CMakeLists.txt"
echo "  - .gitignore"
echo "  - Setup/build scripts"
echo ""
echo "Run: git status"
