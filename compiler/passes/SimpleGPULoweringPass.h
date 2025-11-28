#ifndef SimpleGPULoweringPass_H
#define SimpleGPULoweringPass_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createSimpleGPULoweringPass();
void registerSimpleGPULoweringPass();
}

#endif
