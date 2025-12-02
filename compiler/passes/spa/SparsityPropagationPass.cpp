#include "SPADomain.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

using sparseflow::MatrixSparsity;
using sparseflow::SparsityMap;
using sparseflow::makeDenseRows;
using sparseflow::intersectRows;
using sparseflow::unionRows;

struct SparsityPropagationPass
    : public PassWrapper<SparsityPropagationPass, OperationPass<ModuleOp>> {

  // Required for MLIR 18/19 pass registry
  StringRef getArgument() const final { return "sparseflow-spa"; }

  StringRef getDescription() const final {
    return "SparseFlow Sparsity Propagation Analysis (SPA) pass";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // v0.1: just attach a dense rowmask to every linalg.matmul result.
    module.walk([&](Operation *op) {
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        handleMatmul(matmul);
      }
    });
  }

private:
  SparsityMap S;

  void handleMatmul(linalg::MatmulOp matmul) {
    Value result = matmul.getResult(0);

    auto resultType = mlir::dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || resultType.getRank() < 2)
      return;

    int64_t rows = resultType.getShape()[0];
    if (rows <= 0)
      return;

    // For now: assume all rows can be non-zero
    MatrixSparsity info = makeDenseRows(static_cast<int>(rows));
    S[result] = info;

    attachRowMaskAttr(matmul, info);
  }

  void attachRowMaskAttr(Operation *op, const MatrixSparsity &info) {
    MLIRContext *ctx = op->getContext();
    llvm::SmallVector<Attribute, 8> elems;
    elems.reserve(info.rowMask.size());

    for (std::uint8_t bit : info.rowMask) {
      // Store as i1 integer attributes: 0 or 1
      elems.push_back(IntegerAttr::get(IntegerType::get(ctx, 1),
                                       bit ? 1 : 0));
    }

    auto arrAttr = ArrayAttr::get(ctx, elems);
    op->setAttr("sparseflow.spa_rowmask", arrAttr);
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> createSparsityPropagationPass() {
  return std::make_unique<SparsityPropagationPass>();
}

// Register the pass with MLIR
static PassRegistration<SparsityPropagationPass> regSparseFlowSPA;
