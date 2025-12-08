#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

using namespace mlir;

namespace {

struct SPAExportPass : public PassWrapper<SPAExportPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SPAExportPass)
  
  StringRef getArgument() const final { return "sparseflow-spa-export"; }
  StringRef getDescription() const final {
    return "Export SPA sparsity info to JSON";
  }
  
  void runOnOperation() override {
    llvm::json::Object root;
    llvm::json::Array operations;
    
    int opCount = 0;
    int sparseOps = 0;
    
    getOperation().walk([&](Operation *op) {
      if (!op->hasAttr("sparseflow.spa_rowmask") && 
          !op->hasAttr("sparseflow.spa_colmask")) {
        return;
      }
      
      sparseOps++;
      llvm::json::Object opObj;
      
      opObj["name"] = op->getName().getStringRef().str();
      opObj["id"] = opCount++;
      
      if (auto rowArr = op->getAttrOfType<ArrayAttr>("sparseflow.spa_rowmask")) {
        llvm::json::Array rowMask;
        int zeroRows = 0;
        for (auto attr : rowArr) {
          bool val = cast<BoolAttr>(attr).getValue();
          rowMask.push_back(val);
          if (!val) zeroRows++;
        }
        opObj["rowmask"] = std::move(rowMask);
        opObj["zero_rows"] = zeroRows;
        opObj["total_rows"] = (int)rowArr.size();
        if (rowArr.size() > 0) {
          opObj["row_sparsity_pct"] = (zeroRows * 100.0) / rowArr.size();
        }
      }
      
      if (auto colArr = op->getAttrOfType<ArrayAttr>("sparseflow.spa_colmask")) {
        llvm::json::Array colMask;
        int zeroCols = 0;
        for (auto attr : colArr) {
          bool val = cast<BoolAttr>(attr).getValue();
          colMask.push_back(val);
          if (!val) zeroCols++;
        }
        opObj["colmask"] = std::move(colMask);
        opObj["zero_cols"] = zeroCols;
        opObj["total_cols"] = (int)colArr.size();
        if (colArr.size() > 0) {
          opObj["col_sparsity_pct"] = (zeroCols * 100.0) / colArr.size();
        }
      }
      
      operations.push_back(std::move(opObj));
    });
    
    root["operations"] = std::move(operations);
    root["total_operations"] = opCount;
    root["sparse_operations"] = sparseOps;
    
    std::string jsonStr;
    llvm::raw_string_ostream os(jsonStr);
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
    
    std::ofstream outFile("spa_sparsity.json");
    outFile << jsonStr;
    outFile.close();
    
    llvm::outs() << "✅ Exported sparsity info to spa_sparsity.json\n";
    llvm::outs() << "   Total operations: " << opCount << "\n";
    llvm::outs() << "   Sparse operations: " << sparseOps << "\n";
  }
};

} // namespace

namespace {
inline PassRegistration<SPAExportPass> registerPass;
}

void registerSPAExportPass() {
  PassRegistration<SPAExportPass>();
}