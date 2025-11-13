#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {
struct AnnotateNmPass : public PassWrapper<AnnotateNmPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateNmPass)

  // Required for MLIR 19 when using Pass::Option
  AnnotateNmPass() = default;
  AnnotateNmPass(const AnnotateNmPass &other) : PassWrapper(other) {}

  Option<int64_t> mOpt{*this, "m", llvm::cl::desc("N:M sparsity pattern - m value"), llvm::cl::init(4)};
  Option<int64_t> nOpt{*this, "n", llvm::cl::desc("N:M sparsity pattern - n value"), llvm::cl::init(2)};

  StringRef getArgument() const final { return "sparseflow-annotate-nm"; }
  StringRef getDescription() const final { return "Annotate linalg.matmul ops with sparseflow.nm pattern"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    int64_t m = mOpt;
    int64_t n = nOpt;

    if (m <= 0 || n <= 0) {
      func.emitError() << "Invalid N:M pattern: m=" << m << " n=" << n << " (must be positive)";
      signalPassFailure();
      return;
    }

    func.walk([&](linalg::MatmulOp matmulOp) {
      if (matmulOp->hasAttr("sparseflow.nm"))
        return;

      MLIRContext *context = matmulOp.getContext();
      NamedAttrList attrs;
      attrs.append("n", IntegerAttr::get(IntegerType::get(context, 64), n));
      attrs.append("m", IntegerAttr::get(IntegerType::get(context, 64), m));

      matmulOp->setAttr("sparseflow.nm", DictionaryAttr::get(context, attrs));
      matmulOp.emitRemark() << "Annotated with sparseflow.nm = \"" << n << ":" << m << "\"";
    });
  }
};
} // namespace

void registerAnnotateNmPass() {
  static mlir::PassRegistration<AnnotateNmPass> reg;
}
