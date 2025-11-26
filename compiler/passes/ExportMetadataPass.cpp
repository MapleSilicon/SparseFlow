#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace func = mlir::func;

namespace {

struct ExportMetadataPass
  : public PassWrapper<ExportMetadataPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportMetadataPass)

  StringRef getArgument() const final { return "sparseflow-export-metadata"; }
  StringRef getDescription() const final {
    return "Export N:M metadata for linalg.matmul ops as JSON";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    llvm::json::Array matmuls;

    funcOp.walk([&](linalg::MatmulOp mm) {
      auto nAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.n");
      auto mAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.m");
      if (!nAttr || !mAttr)
        return;

      llvm::json::Object obj;
      obj["type"] = "matmul";
      obj["n"] = nAttr.getInt();
      obj["m"] = mAttr.getInt();

      matmuls.push_back(std::move(obj));
    });

    llvm::json::Object root;
    root["matmuls"] = std::move(matmuls);

    llvm::outs() << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
  }
};

} // namespace

void registerExportMetadataPass() {
  PassRegistration<ExportMetadataPass>();
}
