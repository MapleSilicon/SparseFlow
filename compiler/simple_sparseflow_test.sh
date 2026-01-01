#!/usr/bin/env bash
set -euo pipefail

echo "=== Building SparseFlowPasses (single job) ==="
cd ~/src/SparseFlow/compiler/build

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm

ninja -j1 SparseFlowPasses

echo "=== Running SparseFlow pipeline on sparseflow-large-matmul.mlir (defaults n=2, m=4) ==="
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  --pass-pipeline="builtin.module(func.func(sparseflow-annotate-nm,sparseflow-export-metadata,sparseflow-flop-counter))" \
  ../../test/sparseflow-large-matmul.mlir > out_with_ir.mlir

echo "=== Extracting JSON metadata only (defaults n=2, m=4) ==="
mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so \
  --pass-pipeline="builtin.module(func.func(sparseflow-annotate-nm,sparseflow-export-metadata))" \
  ../../test/sparseflow-large-matmul.mlir > out.json

echo "=== Done ==="
echo "JSON metadata:    ~/src/SparseFlow/compiler/build/out.json"
echo "Annotated IR+log: ~/src/SparseFlow/compiler/build/out_with_ir.mlir"
