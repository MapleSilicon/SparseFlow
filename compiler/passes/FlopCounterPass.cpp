#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace func = mlir::func;

namespace {

struct FlopCounterPass
  : public PassWrapper<FlopCounterPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlopCounterPass)

  StringRef getArgument() const final { return "sparseflow-flop-counter"; }
  StringRef getDescription() const final {
    return "Count FLOPs for linalg.matmul ops";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    int64_t totalFlops = 0;

    funcOp.walk([&](linalg::MatmulOp mm) {
      auto aType = mlir::dyn_cast<ShapedType>(mm.getInputs()[0].getType());
      auto bType = mlir::dyn_cast<ShapedType>(mm.getInputs()[1].getType());
      if (!aType || !bType || aType.getRank() != 2 || bType.getRank() != 2)
        return;

      int64_t M = aType.getShape()[0];
      int64_t K = aType.getShape()[1];
      int64_t N = bType.getShape()[1];

      totalFlops += 2 * M * N * K;
    });

    llvm::outs() << "Total FLOPs: " << totalFlops << "\n";
  }
};

} // namespace

void registerFlopCounterPass() {
  PassRegistration<FlopCounterPass>();
}
