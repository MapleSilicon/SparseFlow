#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
namespace func = mlir::func;

namespace {

struct AnnotateNmPass
  : public PassWrapper<AnnotateNmPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateNmPass)

  StringRef getArgument() const final { return "sparseflow-annotate-nm"; }
  StringRef getDescription() const final {
    return "Annotate linalg.matmul ops with sparseflow.n and sparseflow.m attributes";
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    func::FuncOp funcOp = getOperation();

    // Fixed values for N:M sparsity
    int n = 2;
    int m = 4;

    funcOp.walk([&](linalg::MatmulOp op) {
      op->setAttr("sparseflow.n",
                  IntegerAttr::get(IntegerType::get(ctx, 32), n));
      op->setAttr("sparseflow.m",
                  IntegerAttr::get(IntegerType::get(ctx, 32), m));
    });
  }
};

} // namespace

void registerAnnotateNmPass() {
  PassRegistration<AnnotateNmPass>();
}
