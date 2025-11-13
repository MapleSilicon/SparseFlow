#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
using namespace mlir;

namespace {
struct MatmulProbePass
  : public PassWrapper<MatmulProbePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulProbePass)

  StringRef getArgument() const final { return "sparseflow-matmul-probe"; }
  StringRef getDescription() const final {
    return "Report locations of linalg.matmul (first hook for sparsification).";
  }

  void runOnOperation() override {
    auto func = getOperation();
    func.walk([&](linalg::MatmulOp mm){
      auto loc = mm.getLoc();
      func.emitRemark() << "Found linalg.matmul at " << loc;
    });
  }
};
} // namespace

std::unique_ptr<Pass> createMatmulProbePass() {
  return std::make_unique<MatmulProbePass>();
}

// Static registration only (no PassPlugin.h needed)
namespace { PassRegistration<MatmulProbePass> reg; }
