#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct NmConsumerPass : public PassWrapper<NmConsumerPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NmConsumerPass)

  // Required for MLIR 19 when using Pass::Option (even if none are defined)
  NmConsumerPass() = default;
  NmConsumerPass(const NmConsumerPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "sparseflow-nm-consumer"; }
  StringRef getDescription() const final { return "Consume sparseflow.nm annotations and emit diagnostics"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    bool foundAny = false;

    func.walk([&](linalg::MatmulOp matmulOp) {
      auto nmDict = matmulOp->getAttrOfType<DictionaryAttr>("sparseflow.nm");
      if (!nmDict) return;

      foundAny = true;
      auto nAttr = mlir::dyn_cast_or_null<IntegerAttr>(nmDict.get("n"));
      auto mAttr = mlir::dyn_cast_or_null<IntegerAttr>(nmDict.get("m"));
      if (!nAttr || !mAttr) return;

      int64_t n = nAttr.getInt();
      int64_t m = mAttr.getInt();
      if (n <= 0 || m <= 0) return;

      double sparsityRatio = static_cast<double>(n) / static_cast<double>(m);
      double computeReduction = (1.0 - sparsityRatio) * 100.0;

      matmulOp.emitRemark()
        << "N:M pattern " << n << ":" << m
        << " - expected compute reduction: " << computeReduction << "%";
    });

    if (!foundAny)
      func.emitRemark() << "No sparseflow.nm annotations found";
  }
};
} // namespace

void registerSparseflowNmConsumerPass() {
  static mlir::PassRegistration<NmConsumerPass> reg;
}
