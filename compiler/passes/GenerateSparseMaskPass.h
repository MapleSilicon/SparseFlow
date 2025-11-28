#ifndef GenerateSparseMaskPass_H
#define GenerateSparseMaskPass_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createGenerateSparseMaskPass();
void registerGenerateSparseMaskPass();
}

#endif
