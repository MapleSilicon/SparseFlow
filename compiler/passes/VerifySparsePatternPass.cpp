#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::linalg;

namespace {

struct VerifySparsePatternPass
    : public PassWrapper<VerifySparsePatternPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifySparsePatternPass)

  StringRef getArgument() const final { return "sparseflow-verify-pattern"; }
  StringRef getDescription() const final {
    return "Verify that sparseflow.nm annotations are valid for the operation";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    bool failed = false;

    func.walk([&](linalg::MatmulOp mm) {
      // Check for the presence of the sparseflow.nm attribute
      auto dict = mm->getAttrOfType<DictionaryAttr>("sparseflow.nm");
      if (!dict) {
        mm.emitWarning() << "missing sparseflow.nm attribute";
        return;
      }

      auto mAttr = dict.get("m").dyn_cast_or_null<IntegerAttr>();
      auto nAttr = dict.get("n").dyn_cast_or_null<IntegerAttr>();

      if (!mAttr || !nAttr) {
        mm.emitError() << "sparseflow.nm must contain 'm' and 'n' integer attributes";
        failed = true;
        return;
      }

      int64_t m = mAttr.getInt();
      int64_t n = nAttr.getInt();

      // Validate N:M pattern constraints
      if (n <= 0 || m <= 0) {
        mm.emitError() << "N:M pattern values must be positive, got " << n << ":" << m;
        failed = true;
        return;
      }

      if (n > m) {
        mm.emitError() << "N must be <= M in N:M sparsity, got " << n << ":" << m;
        failed = true;
        return;
      }

      // Check tensor dimensions compatibility
      auto outputType = mm.getOutputs()[0].getType().dyn_cast<RankedTensorType>();
      if (!outputType) {
        mm.emitWarning() << "cannot verify pattern for unranked output tensor";
        return;
      }

      // For matmul, check if dimensions are compatible with the pattern
      // A: [M, K], B: [K, N], C: [M, N]
      // We typically apply N:M sparsity to weight matrices (B)
      auto inputTypeA = mm.getInputs()[0].getType().dyn_cast<RankedTensorType>();
      auto inputTypeB = mm.getInputs()[1].getType().dyn_cast<RankedTensorType>();

      if (inputTypeA && inputTypeB) {
        int64_t kDim = inputTypeA.getDimSize(1); // K dimension from A[M, K]
        
        // Check if K dimension is compatible with the pattern block size
        if (kDim % m != 0) {
          mm.emitRemark() << "weight matrix dimension " << kDim 
                         << " may not be optimally tiled for " << n << ":" << m 
                         << " sparsity (not divisible by " << m << ")";
        }
      }

      mm.emitRemark() << "validated " << n << ":" << m << " sparsity pattern";
    });

    if (failed)
      signalPassFailure();
  }
};

} // end anonymous namespace

// Registration hook called from the plugin
void registerVerifySparsePatternPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<VerifySparsePatternPass>();
  });
}
