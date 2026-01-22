#!/bin/bash
set -e

echo "=========================================="
echo "  SparseFlow Correctness Validation"
echo "=========================================="
echo ""
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CUDA: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Run validation
python /workspace/Sparseflow/tests/test_sparse_correctness.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ VALIDATION PASSED - Ready for production"
    exit 0
else
    echo ""
    echo "❌ VALIDATION FAILED - Do not deploy"
    exit 1
fi
