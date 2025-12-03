#!/bin/bash
# Simple wrapper for SparseFlow SPA
PLUGIN="$HOME/src/SparseFlow/compiler/build/passes/SparseFlowPasses.so"
mlir-opt-19 --load-pass-plugin="$PLUGIN" \
  --sparseflow-spa \
  --sparseflow-spa-export \
  "$@"
