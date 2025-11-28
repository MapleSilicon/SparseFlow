#ifndef FLOP_COUNTER_PASS_H
#define FLOP_COUNTER_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<mlir::Pass> createFlopCounterPass();
void registerFlopCounterPass();

} // namespace mlir

#endif