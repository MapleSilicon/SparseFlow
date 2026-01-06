#!/bin/bash
set -e

echo "🔧 Installing SparseFlow CLI tools..."

# Create bin directory
mkdir -p ~/.local/bin

# Install tools
for tool in audit convert benchmark; do
    echo "  Installing sparseflow-$tool..."
    cp tools/$tool/sparseflow_$tool.py ~/.local/bin/sparseflow-$tool
    chmod +x ~/.local/bin/sparseflow-$tool
done

echo ""
echo "✅ Tools installed to ~/.local/bin/"
echo ""
echo "Add to PATH (if not already):"
echo '  export PATH="$HOME/.local/bin:$PATH"'
echo ""
echo "Usage:"
echo "  sparseflow-audit --model llama-7b --qps 1000"
echo "  sparseflow-convert --input model.pt --output model.sf"
echo "  sparseflow-benchmark --size 4096x4096"
echo ""
