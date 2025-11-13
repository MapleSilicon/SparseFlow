#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace {

struct SimpleGPULoweringPass
    : public PassWrapper<SimpleGPULoweringPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimpleGPULoweringPass)

  StringRef getArgument() const final {
    return "sparseflow-simple-gpu-lowering";
  }

  StringRef getDescription() const final {
    return "Simple GPU lowering placeholder for SparseFlow";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](linalg::MatmulOp mm) {
      mm.emitRemark() << "GPU lowering would be applied here";
    });
  }
};

} // namespace

void registerSimpleGPULoweringPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<SimpleGPULoweringPass>();
  });
}
