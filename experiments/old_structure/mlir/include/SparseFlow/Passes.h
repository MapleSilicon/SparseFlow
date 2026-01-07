//===- Passes.h - SparseFlow optimization passes ----------------*- C++ -*-===//

#ifndef SPARSEFLOW_PASSES_H
#define SPARSEFLOW_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace sparseflow {

/// Create a pass that optimizes tile sizes based on GPU architecture
std::unique_ptr<Pass> createTileSizeOptimizationPass();

/// Create a pass that fuses compatible operations
std::unique_ptr<Pass> createOperationFusionPass();

/// Create a pass that lowers SparseFlow dialect to LLVM/CUDA
std::unique_ptr<Pass> createLowerToGPUPass();

/// Register all SparseFlow passes
void registerSparseFlowPasses();

} // namespace sparseflow
} // namespace mlir

#endif // SPARSEFLOW_PASSES_H
