#include "SparseFlowDialect.h"
#include "SparseFlow/SparseFlowOps.h"

#include "SparseFlow/SparseFlowOpsDialect.cpp.inc"

using namespace mlir;
using namespace sparseflow;

SparseFlowDialect::SparseFlowDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<SparseFlowDialect>()) {
  initialize();
}

void sparseflow::SparseFlowDialect::initialize() {
  addOperations
#define GET_OP_LIST
#include "SparseFlow/SparseFlowOps.cpp.inc"
  >();
}

#define GET_OP_CLASSES
#include "SparseFlow/SparseFlowOps.cpp.inc"
