#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace {
struct LowerToSparseflowCallPass
  : public PassWrapper<LowerToSparseflowCallPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToSparseflowCallPass)

  StringRef getArgument() const final { return "sparseflow-lower-to-call"; }
  StringRef getDescription() const final {
    return "Lower linalg.matmul with sparseflow.nm to func.call @sparseflow.matmul_nm";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder b(func.getContext());
    SmallVector<Operation*> toErase;

    func.walk([&](linalg::MatmulOp mm) {
      if (!mm->hasAttr("sparseflow.nm")) return;

      Location loc = mm.getLoc();
      Value A = mm.getInputs()[0];
      Value B = mm.getInputs()[1];
      Value out = mm.getOutputs()[0];
      auto resTy = out.getType().cast<RankedTensorType>();

      // Ensure callee decl exists in the parent module.
      ModuleOp mod = func->getParentOfType<ModuleOp>();
      StringRef callee = "sparseflow.matmul_nm";
      if (!mod.lookupSymbol<func::FuncOp>(callee)) {
        OpBuilder modB(mod.getBodyRegion());
        auto fnTy = FunctionType::get(mod.getContext(),
                                      {A.getType(), B.getType()}, {resTy});
        modB.create<func::FuncOp>(loc, callee, fnTy);
      }

      b.setInsertionPoint(mm);
      auto call = b.create<func::CallOp>(loc, callee, TypeRange{resTy},
                                         ValueRange{A, B});
      mm.getResult(0).replaceAllUsesWith(call.getResult(0));
      toErase.push_back(mm);
    });

    for (Operation *op : toErase)
      op->erase();
  }
};
} // namespace

// Static registration so mlir-opt sees the pass when the plugin loads.
namespace {
struct _Reg {
  _Reg() { PassRegistration<LowerToSparseflowCallPass>(); }
} _reg;

void registerSparseflowLowerToSparseflowCallPass() {
  static PassRegistration<LowerToSparseflowCall> pass;
}
