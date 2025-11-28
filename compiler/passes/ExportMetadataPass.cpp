#include "ExportMetadataPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct ExportMetadataPass
  : public PassWrapper<ExportMetadataPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportMetadataPass)

  StringRef getArgument() const final { return "sparseflow-export-metadata"; }
  StringRef getDescription() const final {
    return "Export N:M metadata and dimensions for linalg.matmul ops as JSON";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::json::Array matmuls;

    module.walk([&](linalg::MatmulOp mm) {
      auto nAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.n");
      auto mAttr = mm->getAttrOfType<IntegerAttr>("sparseflow.m");
      
      // Use defaults if attributes not found
      int n = nAttr ? nAttr.getInt() : 2;
      int m = mAttr ? mAttr.getInt() : 4;

      // Extract matrix dimensions from the matmul operation
      int64_t dimM = -1, dimN = -1, dimK = -1;
      
      // Input A: [M x K], Input B: [K x N], Output C: [M x N]
      auto inputAType = mm.getInputs()[0].getType().dyn_cast<ShapedType>();
      auto inputBType = mm.getInputs()[1].getType().dyn_cast<ShapedType>();
      
      if (inputAType && inputAType.hasRank() && inputAType.getRank() == 2) {
        dimM = inputAType.getDimSize(0); // Rows of A
        dimK = inputAType.getDimSize(1); // Columns of A = Rows of B
      }
      
      if (inputBType && inputBType.hasRank() && inputBType.getRank() == 2) {
        dimN = inputBType.getDimSize(1); // Columns of B
      }

      llvm::json::Object obj;
      obj["type"] = "matmul";
      obj["n"] = n;
      obj["m"] = m;
      
      // Add dimensions - use extracted values or defaults
      obj["M"] = (dimM != -1) ? dimM : 64;
      obj["N"] = (dimN != -1) ? dimN : 64;  
      obj["K"] = (dimK != -1) ? dimK : 64;

      matmuls.push_back(std::move(obj));
      
      // Debug output
      llvm::errs() << "Exporting matmul: " << dimM << "x" << dimK 
                   << " * " << dimK << "x" << dimN 
                   << " with " << n << ":" << m << " sparsity\n";
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
