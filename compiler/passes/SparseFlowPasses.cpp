#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "sparseflow/Passes.h" // Includes the function prototypes

// Forward declaration from PassRegistration.cpp
void registerAllSparseFlowPasses(); 

// This function is the main entry point for the shared object/plugin.
extern "C" LLVM_ATTRIBUTE_WEAK void registerSparseFlowPasses() {
  // Call the utility function that registers all individual passes.
  registerAllSparseFlowPasses();
}
