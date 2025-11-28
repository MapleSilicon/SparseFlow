#ifndef ANNOTATE_NM_PASS_H
#define ANNOTATE_NM_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createAnnotateNmPass();
void registerAnnotateNmPass();
} // namespace mlir

#endif
