#!/bin/bash
set -e

echo "🔧 Installing SparseFlow Python package..."

# Check PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "❌ PyTorch not found. Install with:"
    echo "   pip install torch"
    exit 1
}

# Check CUDA
python3 -c "import torch; assert torch.cuda.is_available()" || {
    echo "⚠️  CUDA not available. SparseFlow requires CUDA."
    exit 1
}

# Install in development mode
pip install -e .

echo "✅ SparseFlow installed!"
echo ""
echo "Test it:"
echo "  python3 -c 'import sparseflow; print(sparseflow.check_sparse_support())'"
echo ""
