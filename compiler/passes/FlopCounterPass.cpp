#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace {

struct FlopCounterPass
    : public PassWrapper<FlopCounterPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlopCounterPass)

  StringRef getArgument() const final { return "sparseflow-flop-counter"; }

  StringRef getDescription() const final {
    return "Count dense vs sparse FLOPs for linalg.matmul (2:4 pattern)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    int64_t denseFlops = 0;
    int64_t sparseFlops = 0;

    func.walk([&](linalg::MatmulOp mm) {
      auto lhsType = dyn_cast<RankedTensorType>(mm.getInputs()[0].getType());
      auto rhsType = dyn_cast<RankedTensorType>(mm.getInputs()[1].getType());
      if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
          rhsType.getRank() != 2)
        return;

      int64_t M = lhsType.getDimSize(0);
      int64_t K = lhsType.getDimSize(1);
      int64_t N = rhsType.getDimSize(1);

      if (M < 0 || N < 0 || K < 0)
        return;

      // Dense matmul FLOPs ~ 2 * M * K * N (mul + add)
      int64_t dense = 2 * M * K * N;
      denseFlops += dense;

      // For 2:4 sparsity on K dim, keep half of K positions.
      int64_t effectiveK = K / 2;
      int64_t sparse = 2 * M * effectiveK * N;
      sparseFlops += sparse;
    });

    if (denseFlops == 0)
      return;

    int64_t saved = denseFlops - sparseFlops;
    double savingsPct =
        100.0 * static_cast<double>(saved) / static_cast<double>(denseFlops);

    func.emitRemark()
        << "SparseFlow FLOPs (matmul only): dense=" << denseFlops
        << " sparse=" << sparseFlops
        << " saved=" << saved
        << " (" << savingsPct << "%)";
  }
};

} // namespace

void registerFlopCounterPass() {
  PassRegistration<FlopCounterPass>();
}