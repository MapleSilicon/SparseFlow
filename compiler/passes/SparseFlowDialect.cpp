#include "SparseFlow/SparseFlowDialect.h"
#include "SparseFlow/SparseFlowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::sparseflow;

#include "SparseFlow/SparseFlowOpsDialect.cpp.inc"

void SparseFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SparseFlow/SparseFlowOps.cpp.inc"
  >();
}

// Register the dialect with MLIR
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparseflow::SparseFlowDialect)
