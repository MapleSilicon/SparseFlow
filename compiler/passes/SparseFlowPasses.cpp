#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/JSON.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace mlir;

// Forward declarations for all pass registrations
namespace mlir {
void registerSparseMatmulRewritePass();
}

namespace {
class AnnotateNmPass : public PassWrapper<AnnotateNmPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateNmPass)

  AnnotateNmPass() = default;
  AnnotateNmPass(const AnnotateNmPass&) = default;

  Option<int> n{*this, "N", llvm::cl::desc("N for N:M sparsity"), llvm::cl::init(2)};
  Option<int> m{*this, "M", llvm::cl::desc("M for N:M sparsity"), llvm::cl::init(4)};

  void runOnOperation() override {
    Operation* module = getOperation();

    // Walk through all operations in the module
    module->walk([&](Operation* op) {
      if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
        // Create a dictionary attribute with n and m
        auto config = DictionaryAttr::get(
            op->getContext(),
            {NamedAttribute(StringAttr::get(op->getContext(), "sparseflow.n"),
                            IntegerAttr::get(IntegerType::get(op->getContext(), 32), n)),
             NamedAttribute(StringAttr::get(op->getContext(), "sparseflow.m"),
                            IntegerAttr::get(IntegerType::get(op->getContext(), 32), m))});
        matmulOp->setAttr("sparseflow.config", config);
      }
    });
  }
};

class ExportMetadataPass : public PassWrapper<ExportMetadataPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportMetadataPass)

  ExportMetadataPass() = default;
  ExportMetadataPass(const ExportMetadataPass&) = default;

  void runOnOperation() override {
    Operation* module = getOperation();
    llvm::json::Array matmuls;

    module->walk([&](Operation* op) {
      if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
        auto inputs = matmulOp.getInputs();
        if (inputs.size() < 2) {
          return;
        }

        auto A = inputs[0].getType().dyn_cast<RankedTensorType>();
        auto B = inputs[1].getType().dyn_cast<RankedTensorType>();
        if (!A || !B) {
          return;
        }

        int M = A.getShape()[0];
        int K = A.getShape()[1];
        int N = B.getShape()[1];

        int n_val = 2;
        int m_val = 4;

        // Check if the operation has the sparseflow.config attribute
        auto config = matmulOp->getAttrOfType<DictionaryAttr>("sparseflow.config");
        if (config) {
          if (auto n_attr = config.get("sparseflow.n").dyn_cast<IntegerAttr>()) {
            n_val = n_attr.getValue().getZExtValue();
          }
          if (auto m_attr = config.get("sparseflow.m").dyn_cast<IntegerAttr>()) {
            m_val = m_attr.getValue().getZExtValue();
          }
        }

        // Derived metrics
        long long totalMACs = 1ll * M * N * K;
        double activeFraction =
            (m_val > 0) ? static_cast<double>(n_val) / static_cast<double>(m_val) : 0.0;
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

        matmuls.push_back(std::move(matmulInfo));

        // Log what we found
        llvm::outs() << "Exporting matmul: " << M << "x" << K << " * " << K << "x" << N
                     << " with " << n_val << ":" << m_val << " sparsity\n"
                     << "  totalMACs=" << totalMACs
                     << " executedMACs=" << executedMACs
                     << " density=" << activeFraction
                     << " speedup=" << speedup << "x\n";
      }
    });

    llvm::json::Object topLevel{
      {"matmuls", std::move(matmuls)}
    };

    std::string jsonStr;
    llvm::raw_string_ostream os(jsonStr);
    os << llvm::json::Value(std::move(topLevel));
    os.flush();

    std::error_code EC;
    llvm::raw_fd_ostream out("hardware_config.json", EC, llvm::sys::fs::OF_Text);
    if (EC) {
      module->emitError() << "Failed to open hardware_config.json: " << EC.message();
      return signalPassFailure();
    }
    out << jsonStr;
    out.flush();
  }
};
} // namespace

void registerSparseFlowPasses() {
  PassRegistration<AnnotateNmPass>(
      "sparseflow-annotate-nm",
      "Annotate linalg.matmul operations with N:M sparsity configuration");

  PassRegistration<ExportMetadataPass>(
      "sparseflow-export-metadata",
      "Export hardware configuration metadata to JSON");

  // Register the sparse matmul rewrite pass
  mlir::registerSparseMatmulRewritePass();
}