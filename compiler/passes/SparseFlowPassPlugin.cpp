//===- SparseFlowPassPlugin.cpp - SparseFlow Pass Plugin Registration ----===//
//
// SparseFlow MLIR Pass Plugin
// All passes use static PassRegistration - no explicit calls needed
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassRegistry.h"

// Plugin entry point - MLIR loads this automatically
// All SparseFlow passes use static PassRegistration<> so they
// register themselves when the .so is loaded
extern "C" LLVM_ATTRIBUTE_WEAK void registerPasses() {
  // Static registrations handle everything - no explicit calls needed
}
