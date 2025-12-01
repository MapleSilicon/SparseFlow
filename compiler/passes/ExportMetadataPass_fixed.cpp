#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

using namespace mlir;

namespace {

struct ExportMetadataPass : public PassWrapper<ExportMetadataPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    
    llvm::json::Array operations;
    
    func.walk([&](Operation *op) {
      if (auto mm = dyn_cast<linalg::MatmulOp>(op)) {
        
        // Get input types - FIXED: use mlir::dyn_cast
        auto inputAType = mlir::dyn_cast<ShapedType>(mm.getInputs()[0].getType());
        auto inputBType = mlir::dyn_cast<ShapedType>(mm.getInputs()[1].getType());
        
        if (!inputAType || !inputBType) return;
        
        auto shapeA = inputAType.getShape();
        auto shapeB = inputBType.getShape();
        
        if (shapeA.size() != 2 || shapeB.size() != 2) return;
        
        int64_t M = shapeA[0];
        int64_t K = shapeA[1];
        int64_t N = shapeB[1];
        
        // Get sparsity config
        int n_val = 2, m_val = 4;
        if (auto config = mm->getAttrOfType<DictionaryAttr>("sparseflow.config")) {
          if (auto n_attr = mlir::dyn_cast<IntegerAttr>(config.get("sparseflow.n"))) {
            n_val = n_attr.getInt();
          }
          if (auto m_attr = mlir::dyn_cast<IntegerAttr>(config.get("sparseflow.m"))) {
            m_val = m_attr.getInt();
          }
        }
        
        // Compute metrics
        long long totalMACs = 1ll * M * N * K;
        double activeFraction = (m_val > 0) ? static_cast<double>(n_val) / static_cast<double>(m_val) : 0.0;
        long long executedMACs = static_cast<long long>(totalMACs * activeFraction);
        double speedup = (activeFraction > 0.0) ? (1.0 / activeFraction) : 0.0;
        
        llvm::json::Object matmulInfo{
          {"type", "matmul"},
          {"M", M},
          {"N", N},
          {"K", K},
          {"n", n_val},
          {"m", m_val},
          {"totalMACs", totalMACs},
          {"executedMACs", executedMACs},
          {"density", activeFraction},
          {"theoreticalSpeedup", speedup}
        };
        
        operations.push_back(std::move(matmulInfo));
      }
    });
    
    // Export JSON
    llvm::json::Object root;
    root["version"] = "1.0";
    root["operations"] = std::move(operations);
    
    std::error_code ec;
    llvm::raw_fd_ostream out("hardware_config.json", ec);
    if (!ec) {
      out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    }
  }
  
  StringRef getArgument() const final { return "sparseflow-export-metadata"; }
  StringRef getDescription() const final {
    return "Export N:M metadata and dimensions for linalg.matmul ops as JSON";
  }
};

} // namespace

namespace mlir {
void registerExportMetadataPass() {
  PassRegistration<ExportMetadataPass>();
}

std::unique_ptr<Pass> createExportMetadataPass() {
  return std::make_unique<ExportMetadataPass>();
}
} // namespace mlir
