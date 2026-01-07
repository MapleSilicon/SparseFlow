//===- PassRegistration.cpp - Register SparseFlow passes ------------------===//

#include "SparseFlow/Passes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sparseflow {

void registerSparseFlowPasses() {
    // Register all passes
    PassRegistration<TileSizeOptimizationPass>();
    PassRegistration<OperationFusionPass>();
    
    // TODO: Add more passes
}

} // namespace sparseflow
} // namespace mlir
