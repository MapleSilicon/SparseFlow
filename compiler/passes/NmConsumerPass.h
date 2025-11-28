#ifndef NmConsumerPass_H
#define NmConsumerPass_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createNmConsumerPass();
void registerNmConsumerPass();
}

#endif
