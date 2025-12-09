//===- SparseMatmulRewritePass.cpp ---------------------------------------===//
//
// Rewrite SPA-marked linalg.matmul operations into calls to sparseflow
// runtime kernel with N:M pattern support.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include <string>

using namespace mlir;

namespace {

struct SparseMatmulRewritePass
    : public PassWrapper<SparseMatmulRewritePass,
                         OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseMatmulRewritePass)
  
  StringRef getArgument() const override { return "sparseflow-rewrite-matmul"; }
  
  StringRef getDescription() const override {
    return "Rewrite SPA-marked matmuls into N:M sparse kernel calls";
  }
  
  // Determine the function name based on N:M pattern
  std::string getFunctionName(Operation *op) {
    // Check for N:M pattern attributes
    if (auto nAttr = op->getAttrOfType<IntegerAttr>("sparseflow.nm_n")) {
      if (auto mAttr = op->getAttrOfType<IntegerAttr>("sparseflow.nm_m")) {
        int N = nAttr.getInt();
        int M = mAttr.getInt();
        
        // Generate function name: sparse_matmul_N_M
        return "sparse_matmul_" + std::to_string(N) + "_" + std::to_string(M);
      }
    }
    
    // Default: 2:4 pattern (backwards compatibility)
    return "sparse_matmul_2_4";
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
      
      // Determine which function to call based on N:M pattern
      std::string fnName = getFunctionName(mm);
      
      // Get or create runtime function declaration
      func::FuncOp kernel = module.lookupSymbol<func::FuncOp>(fnName);
      
      if (!kernel) {
        OpBuilder mb(module.getBodyRegion());
        mb.setInsertionPointToStart(&module.getBodyRegion().front());
        
        auto fnType = mb.getFunctionType({lhsType, rhsType}, {outType});
        kernel = mb.create<func::FuncOp>(mm.getLoc(), fnName, fnType);
        kernel.setPrivate();
        
        // Add N:M pattern info as attributes on the kernel
        if (auto nAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.nm_n")) {
          kernel->setAttr("sparseflow.nm_n", nAttr);
        }
        if (auto mAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.nm_m")) {
          kernel->setAttr("sparseflow.nm_m", mAttr);
        }
        if (auto dirAttr = mm->getAttrOfType<StringAttr>("sparseflow.nm_direction")) {
          kernel->setAttr("sparseflow.nm_direction", dirAttr);
        }
      }
      
      // Rewrite matmul to call the appropriate kernel
      builder.setInsertionPoint(mm);
      auto call = builder.create<func::CallOp>(
          mm.getLoc(), kernel, ValueRange{lhs, rhs});
      
      // Transfer N:M pattern attributes to the call
      if (auto nAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.nm_n")) {
        call->setAttr("sparseflow.nm_n", nAttr);
      }
      if (auto mAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.nm_m")) {
        call->setAttr("sparseflow.nm_m", mAttr);
      }
      if (auto patternAttr = mm->getAttrOfType<StringAttr>("sparseflow.nm_pattern")) {
        call->setAttr("sparseflow.nm_pattern", patternAttr);
      }
      
      mm.replaceAllUsesWith(call.getResults());
      mm.erase();
    }
  }
};

} // namespace

namespace {
inline PassRegistration<SparseMatmulRewritePass> registerPass;
}
