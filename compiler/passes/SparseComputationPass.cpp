#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"

using namespace mlir;

namespace {

struct SparseComputationPass
    : public PassWrapper<SparseComputationPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseComputationPass)

  StringRef getArgument() const final {
    return "sparseflow-sparse-compute";
  }

  StringRef getDescription() const final {
    return "Transform dense operations to sparse computation using N:M masks "
           "(placeholder: emits remarks only)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](linalg::MatmulOp mm) {
      // Look for sparseflow.nm attribute: { n = ..., m = ... }
      Attribute nmAttr = mm->getAttr("sparseflow.nm");
      Attribute maskAttr = mm->getAttr("sparseflow.mask");

      if (!nmAttr || !maskAttr) {
        mm.emitRemark()
            << "sparseflow-sparse-compute: missing nm or mask attribute, "
               "skipping sparse lowering";
        return;
      }

      auto dictAttr = nmAttr.dyn_cast_or_null<DictionaryAttr>();
      int64_t nVal = 0;
      int64_t mVal = 0;

      if (dictAttr) {
        if (auto nA = dictAttr.get("n").dyn_cast_or_null<IntegerAttr>())
          nVal = nA.getInt();
        if (auto mA = dictAttr.get("m").dyn_cast_or_null<IntegerAttr>())
          mVal = mA.getInt();
      }

      mm.emitRemark()
          << "sparseflow-sparse-compute: would lower matmul using "
          << nVal << ":" << mVal << " sparse mask";
    });
  }
};

} // namespace

void registerSparseComputationPass() {
  ::mlir::registerPass([]() {
    return std::make_unique<SparseComputationPass>();
  });
}
