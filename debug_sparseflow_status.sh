#!/usr/bin/env bash

echo "=== SparseFlow Debug Status ==="
echo

echo "=== Git status ==="
git status
echo

echo "=== Top-level files ==="
ls -la
echo

echo "=== Compiler directory ==="
ls -la compiler
echo

echo "=== Runtime directory ==="
ls -la runtime
echo

echo "=== Pass build directory ==="
if [ -d compiler/build/passes ]; then
  cd compiler/build
  echo "Now in: $(pwd)"
  echo
  echo "=== List passes/ directory ==="
  ls -la passes
  echo

  echo "=== Find all .so files under passes ==="
  find passes -maxdepth 2 -type f -name '*.so' 2>/dev/null
  echo

  echo "=== Show FlopCounter object ==="
  if [ -f passes/CMakeFiles/SparseFlowPasses.dir/FlopCounterPass.cpp.o ]; then
    ls -la passes/CMakeFiles/SparseFlowPasses.dir/FlopCounterPass.cpp.o
  else
    echo "FlopCounterPass.cpp.o not found"
  fi
  echo

  echo "=== Check for createFlopCounterPass symbol ==="
  if [ -f passes/SparseFlowPasses.so ]; then
    echo "Checking passes/SparseFlowPasses.so"
    nm -C passes/SparseFlowPasses.so 2>/dev/null | grep createFlopCounterPass || echo "symbol not found"
  else
    echo "SparseFlowPasses.so not found"
  fi
  echo

  echo "=== Check for mlirGetPassPluginInfo symbol ==="
  if [ -f passes/SparseFlowPasses.so ]; then
    echo "Checking passes/SparseFlowPasses.so"
    nm -C passes/SparseFlowPasses.so 2>/dev/null | grep mlirGetPassPluginInfo || echo "symbol not found"
  else
    echo "SparseFlowPasses.so not found"
  fi
  echo

  echo "=== Try loading plugin with mlir-opt ==="
  if [ -f passes/SparseFlowPasses.so ]; then
    echo "Attempting: mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so -help"
    mlir-opt-19 -load-pass-plugin ./passes/SparseFlowPasses.so -help 2>&1 | grep -A5 "sparseflow" || echo "No sparseflow passes found in help"
  else
    echo "SparseFlowPasses.so not found"
  fi
  echo

  echo "=== Hardware config file ==="
  if [ -f hardware_config.json ]; then
    echo "hardware_config.json exists:"
    ls -la hardware_config.json
    echo
    echo "Content (first 30 lines):"
    head -n 30 hardware_config.json
  else
    echo "hardware_config.json not found"
  fi
  echo

  cd ~/src/SparseFlow
else
  echo "compiler/build/passes directory not found"
fi

echo
echo "=== Done ==="

chmod +x debug_sparseflow_status.sh

./debug_sparseflow_status.sh