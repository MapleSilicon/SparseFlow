#ifndef VerifySparsePatternPass_H
#define VerifySparsePatternPass_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createVerifySparsePatternPass();
void registerVerifySparsePatternPass();
}

#endif
