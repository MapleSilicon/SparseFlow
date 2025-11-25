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

  FlopCounterPass() = default;
  FlopCounterPass(const FlopCounterPass &) = default;

  StringRef getArgument() const final { return "sparseflow-flop-counter"; }

  StringRef getDescription() const final {
    return "Count dense vs sparse FLOPs for linalg.matmul (2:4 pattern)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    int64_t denseFlops = 0;
    int64_t sparseFlops = 0;

    // Walk through all linalg.matmul operations in the function.
    func.walk([&](linalg::MatmulOp matmulOp) {
      // Get the types of the input matrices.
      Type lhsType = matmulOp.getInputs()[0].getType();
      Type rhsType = matmulOp.getInputs()[1].getType();
      Type resultType = matmulOp.getOutputs()[0].getType();

      // We assume that the matrices are 2D and of static shape.
      auto lhsShape = lhsType.cast<RankedTensorType>().getShape();
      auto rhsShape = rhsType.cast<RankedTensorType>().getShape();
      auto resultShape = resultType.cast<RankedTensorType>().getShape();

      // For a matrix multiplication A (MxK) * B (KxN) -> C (MxN),
      // the number of dense FLOPs is 2 * M * N * K.
      int64_t M = resultShape[0];
      int64_t N = resultShape[1];
      int64_t K = lhsShape[1];

      denseFlops += 2 * M * N * K;

      // For 2:4 sparsity, we assume that the left-hand side (LHS) is sparse.
      // In 2:4 sparsity, every 4 elements have 2 non-zeros, so the number of non-zeros in LHS is (M * K) / 4 * 2 = (M * K) / 2.
      // Then the number of sparse FLOPs is 2 * (number of non-zeros in LHS) * N = 2 * (M * K / 2) * N = M * K * N.
      sparseFlops += M * K * N;
    });

    // Print the results for this function.
    if (denseFlops > 0) {
      double sparsityRatio = 1.0 - (static_cast<double>(sparseFlops) / static_cast<double>(denseFlops));
      func.emitRemark() << "Dense FLOPs: " << denseFlops << ", Sparse FLOPs (2:4): " << sparseFlops
                        << ", Sparsity ratio: " << (sparsityRatio * 100.0) << "%";
    }
  }
};
} // namespace

void registerFlopCounter() {
  static mlir::PassRegistration<FlopCounterPass> reg;
}