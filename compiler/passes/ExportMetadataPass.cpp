#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Location.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::linalg;

namespace {

struct ExportMetadataPass
    : public PassWrapper<ExportMetadataPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportMetadataPass)

  StringRef getArgument() const final { return "sparseflow-export-metadata"; }
  StringRef getDescription() const final {
    return "Export linalg.matmul with sparseflow.nm into JSON metadata";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::json::Array ops;
    int64_t totalOps = 0;

    func.walk([&](linalg::MatmulOp mm) {
      llvm::json::Object opObj;

      // ----- Location -----
      llvm::json::Object locObj;
      if (auto fileLoc = mm.getLoc().dyn_cast<FileLineColLoc>()) {
        locObj["file"] = fileLoc.getFilename().str();
        locObj["line"] = fileLoc.getLine();
        locObj["column"] = fileLoc.getColumn();
      }
      opObj["location"] = std::move(locObj);

      // ----- nm_pattern from sparseflow.nm = {m = ..., n = ...} -----
      std::string nmPattern = "unknown";
      if (auto dict = mm->getAttrOfType<DictionaryAttr>("sparseflow.nm")) {
        auto mAttr = dict.get("m").dyn_cast_or_null<IntegerAttr>();
        auto nAttr = dict.get("n").dyn_cast_or_null<IntegerAttr>();
        if (mAttr && nAttr) {
          nmPattern = (llvm::Twine(nAttr.getInt()) + ":" +
                       llvm::Twine(mAttr.getInt()))
                          .str();
        }
      }
      opObj["nm_pattern"] = nmPattern;

      // ----- op_type -----
      opObj["op_type"] = "linalg.matmul";

      // ----- operands: inputs vs outputs -----
      llvm::json::Array operands;

      // Inputs
      for (Value v : mm.getInputs()) {
        llvm::json::Object o;
        o["role"] = "input";
        // Convert type to string
        std::string typeStr;
        llvm::raw_string_ostream os(typeStr);
        os << v.getType();
        o["type"] = typeStr;
        operands.push_back(std::move(o));
      }

      // Outputs
      for (Value v : mm.getOutputs()) {
        llvm::json::Object o;
        o["role"] = "output";
        // Convert type to string
        std::string typeStr;
        llvm::raw_string_ostream os(typeStr);
        os << v.getType();
        o["type"] = typeStr;
        operands.push_back(std::move(o));
      }

      opObj["operands"] = std::move(operands);

      ops.push_back(std::move(opObj));
      ++totalOps;
    });

    // If there were no matmuls, still print something deterministic
    llvm::json::Object metaObj;
    metaObj["version"] = "1.0";
    metaObj["total_ops"] = totalOps;

    llvm::json::Object root;
    root["metadata"] = std::move(metaObj);
    root["operations"] = std::move(ops);

    // Output the JSON
    llvm::outs() << "SPARSEFLOW_METADATA: ";
    llvm::outs() << llvm::json::Value(std::move(root));
    llvm::outs() << "\n";
  }
};

} // end anonymous namespace

// Registration hook called from the plugin
void registerExportMetadataPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ExportMetadataPass>();
  });
}
