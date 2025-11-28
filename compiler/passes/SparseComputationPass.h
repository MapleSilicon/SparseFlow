#ifndef SparseComputationPass_H
#define SparseComputationPass_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createSparseComputationPass();
void registerSparseComputationPass();
}

#endif
