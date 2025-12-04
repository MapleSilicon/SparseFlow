#ifndef SPARSEFLOW_DIALECT_H
#define SPARSEFLOW_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace sparseflow {

class SparseFlowDialect : public mlir::Dialect {
public:
  explicit SparseFlowDialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "sparseflow"; }

  void initialize();
};

} // namespace sparseflow

#include "SparseFlow/SparseFlowOpsDialect.h.inc"

#endif // SPARSEFLOW_DIALECT_H
