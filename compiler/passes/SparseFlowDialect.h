#ifndef SPARSEFLOW_DIALECT_H
#define SPARSEFLOW_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "SparseFlow/SparseFlowOpsDialect.h.inc"

namespace mlir {
namespace sparseflow {

class SparseFlowDialect : public Dialect {
public:
  explicit SparseFlowDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "sparseflow"; }
  
  /// Initialize the dialect
  void initialize();
};

} // namespace sparseflow
} // namespace mlir

#endif // SPARSEFLOW_DIALECT_H
