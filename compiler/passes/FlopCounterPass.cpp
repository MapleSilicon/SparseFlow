cd ~/src/SparseFlow/compiler/passes

cat > FlopCounterPass.cpp << 'EOF'
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace {

struct FlopCounterPass
    : public PassWrapper<FlopCounterPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlopCounterPass)

  StringRef getArgument() const final { return "sparseflow-flop-counter"; }

  StringRef getDescription() const final {
    return "Count FLOPs in sparse and dense matmul operations";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    int64_t totalFlops = 0;
    int64_t sparseFlops = 0;
    int64_t denseFlops = 0;
    int64_t matmulCount = 0;

    func.walk([&](Operation* op) {
      // Count arithmetic operations
      if (isa<arith::MulFOp>(op) || isa<arith::AddFOp>(op)) {
        totalFlops++;
      }
      
      // Count linalg.matmul operations and estimate their FLOPs
      if (auto mm = dyn_cast<linalg::MatmulOp>(op)) {
        matmulCount++;
        auto lhsType = mm.getInputs()[0].getType().cast<RankedTensorType>();
        auto rhsType = mm.getInputs()[1].getType().cast<RankedTensorType>();
        
        int64_t M = lhsType.getShape()[0];
        int64_t K = lhsType.getShape()[1];
        int64_t N = rhsType.getShape()[1];
        
        // Dense matmul FLOPs: 2*M*N*K (multiply + add for each element)
        denseFlops += 2 * M * N * K;
        
        // Check if it's sparse
        if (mm->getAttr("sparseflow.nm")) {
          // For 2:4 sparsity, we only compute 50% of K dimension
          sparseFlops += 2 * M * N * (K / 2); // 50% reduction
        } else {
          sparseFlops += 2 * M * N * K; // Same as dense
        }
      }
    });

    // Emit results
    func.emitRemark("FLOP Count Analysis:");
    func.emitRemark("Total arithmetic operations: " + std::to_string(totalFlops));
    if (matmulCount > 0) {
      func.emitRemark("Matmul operations found: " + std::to_string(matmulCount));
      func.emitRemark("Dense matmul FLOPs: " + std::to_string(denseFlops));
      func.emitRemark("Sparse matmul FLOPs: " + std::to_string(sparseFlops));
      if (denseFlops > 0) {
        double savings = 100.0 * (1.0 - (double)sparseFlops / denseFlops);
        func.emitRemark("Computation savings: " + std::to_string(savings) + "%");
      }
    }
  }
};

} // namespace

void registerFlopCounterPass() {
  PassRegistration<FlopCounterPass>();
}
EOF