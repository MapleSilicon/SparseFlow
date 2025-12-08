//===- SparseMatmulRewritePass.cpp ---------------------------------------===//
//
// Rewrite SPA-marked linalg.matmul operations into calls to sparseflow
// runtime kernel.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace {

struct SparseMatmulRewritePass
    : public PassWrapper<SparseMatmulRewritePass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseMatmulRewritePass)

  StringRef getArgument() const override { return "sparseflow-rewrite-matmul"; }
  
  StringRef getDescription() const override {
    return "Rewrite SPA-marked matmuls into sparseflow kernel calls";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    ModuleOp module = func->getParentOfType<ModuleOp>();
    
    if (!module) {
      signalPassFailure();
      return;
    }

    SmallVector<linalg::MatmulOp> targets;

    func.walk([&](linalg::MatmulOp mm) {
      bool hasRowMask = mm->hasAttr("sparseflow.spa_rowmask");
      bool hasColMask = mm->hasAttr("sparseflow.spa_colmask");
      
      if (hasRowMask || hasColMask)
        targets.push_back(mm);
    });

    if (targets.empty())
      return;

    OpBuilder builder(func);

    for (auto mm : targets) {
      auto lhs = mm.getInputs()[0];
      auto rhs = mm.getInputs()[1];
      auto result = mm.getResult(0);

      auto lhsType = cast<RankedTensorType>(lhs.getType());
      auto rhsType = cast<RankedTensorType>(rhs.getType());
      auto outType = cast<RankedTensorType>(result.getType());

      // Get or create runtime function
      constexpr StringLiteral fnName("sparse_matmul_2_4");
      func::FuncOp kernel = module.lookupSymbol<func::FuncOp>(fnName);
      
      if (!kernel) {
        OpBuilder mb(module.getBodyRegion());
        mb.setInsertionPointToStart(&module.getBodyRegion().front());
        
        auto fnType = mb.getFunctionType({lhsType, rhsType}, {outType});
        kernel = mb.create<func::FuncOp>(mm.getLoc(), fnName, fnType);
        kernel.setPrivate();
      }

      // Rewrite matmul to call
      builder.setInsertionPoint(mm);
      auto call = builder.create<func::CallOp>(
          mm.getLoc(), kernel, ValueRange{lhs, rhs});
      
      mm.getResult(0).replaceAllUsesWith(call.getResult(0));
      mm.erase();
    }
  }
};

} // namespace

namespace {
inline PassRegistration<SparseMatmulRewritePass> registerPass;
}
